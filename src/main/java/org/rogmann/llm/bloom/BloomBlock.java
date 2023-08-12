package org.rogmann.llm.bloom;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llm.LlmConfigException;
import org.rogmann.llm.LlmExecutor;
import org.rogmann.llm.nn.LayerNorm;
import org.rogmann.llm.nn.Linear;
import org.rogmann.llm.nn.Tensor;
import org.rogmann.llm.nn.TensorProvider;

/**
 * Class to execute a decoder block (e.g. layer) in the BLOOM-model.
 */
public class BloomBlock {
	/** Logger */
	private static final Logger LOG = Logger.getLogger(BloomBlock.class.getName());

	protected final int fHiddenSize;

	protected final int fNumHeads;

	protected final int fLayer;

	private final LayerNorm inputLayerNorm;

	private final float[][] tSelfAttentionQueryKeyValueWeight;
	private final float[] tSelfAttentionQueryKeyValueBias;
	private final float[][] tSelfAttentionDenseWeight;
	private final float[] tSelfAttentionDenseBias;

	private final float[][] tMlpDenseHTo4HWeight;
	private final float[] tMlpDenseHTo4HBias;
	private final float[][] tMlpDense4HToHWeight;
	private final float[] tMlpDense4HToHBias;

	private final BloomAttention attention;

	private final LayerNorm postAttentionLayerNorm;

	private final BloomMLP mlp;

	private final LlmExecutor executor;
	
	/** Interface to read a data-block of a model-file */
	public static interface IsProvider {
		/**
		 * Gets the input-stream to read a data-block of a model-file.
		 * @param idxFile index of data-block
		 * @return input-stream
		 */
		InputStream create(int idxFile);
	}

	public BloomBlock(final int batchSize, final int hiddenSize, final int numHeads,
			int layer, TensorProvider mapTensor, LlmExecutor executor) throws IOException, LlmConfigException {
		fHiddenSize = hiddenSize;
		fNumHeads = numHeads;

		fLayer = layer;
		String prefix = "h." + layer + '.';
		final float[] tInputLayernormWeight = mapTensor.get(prefix + "input_layernorm.weight").t1;
		final float[] tInputLayernormBias = mapTensor.get(prefix + "input_layernorm.bias").t1;
		if (LOG.isLoggable(Level.FINE)) {
			LOG.fine("ie.weight: " + Arrays.toString(Arrays.copyOfRange(tInputLayernormWeight, 0, 5)));
			LOG.fine("ie.bias:   " + Arrays.toString(Arrays.copyOfRange(tInputLayernormBias, 0, 5)));
		}

		tSelfAttentionQueryKeyValueWeight = mapTensor.get(prefix + "self_attention.query_key_value.weight").t2;
		tSelfAttentionQueryKeyValueBias = mapTensor.get(prefix + "self_attention.query_key_value.bias").t1;
		tSelfAttentionDenseWeight = mapTensor.get(prefix + "self_attention.dense.weight").t2;
		tSelfAttentionDenseBias = mapTensor.get(prefix + "self_attention.dense.bias").t1;

		final float[] tPostAttentionLayernormWeight = mapTensor.get(prefix + "post_attention_layernorm.weight").t1;
		final float[] tPostAttentionLayernormBias = mapTensor.get(prefix + "post_attention_layernorm.bias").t1;

		tMlpDenseHTo4HWeight = mapTensor.get(prefix + "mlp.dense_h_to_4h.weight").t2;
		tMlpDenseHTo4HBias = mapTensor.get(prefix + "mlp.dense_h_to_4h.bias").t1;
		tMlpDense4HToHWeight = mapTensor.get(prefix + "mlp.dense_4h_to_h.weight").t2;
		tMlpDense4HToHBias = mapTensor.get(prefix + "mlp.dense_4h_to_h.bias").t1;

		inputLayerNorm = new LayerNorm(1e-5f, tInputLayernormWeight, tInputLayernormBias);

		final Linear queryKeyValue = new Linear(tSelfAttentionQueryKeyValueWeight, tSelfAttentionQueryKeyValueBias, executor);
		final Linear dense = new Linear(tSelfAttentionDenseWeight, tSelfAttentionDenseBias, executor);
		attention = new BloomAttention(hiddenSize, numHeads, queryKeyValue, dense, executor);

		postAttentionLayerNorm = new LayerNorm(1e-5f, tPostAttentionLayernormWeight, tPostAttentionLayernormBias);

		final Linear denseHTo4H = new Linear(tMlpDenseHTo4HWeight, tMlpDenseHTo4HBias, executor);
		final Linear dense4HToH = new Linear(tMlpDense4HToHWeight, tMlpDense4HToHBias, executor);
		mlp = new BloomMLP(denseHTo4H, dense4HToH, executor);
		
		this.executor = executor;
	}

	/**
	 * Computes a BLOOM-block.
	 * @param inputEmbeds input embeddings
	 * @param hiddenStates tensor (batchSize, numSeq, hiddenSize)
	 * @param fusedQkv temporary tensor ([batchSize][numSeq][3 * dimHidden])
	 * @param attentionMask attention mask
	 * @param alibi ALiBi-tensor
	 * @param attentionResidual attention residual
	 * @param output output tensor (batchSize, numSeq, hiddenSize)
	 */
	public void forward(final float[][][] inputEmbeds, final float[][][] hiddenStates,
			final float[][][] fusedQkv,
			final boolean[][][][] attentionMask, final Tensor alibi, final float[][][] attentionResidual,
			final float[][][] output) {
		final int batchSize = inputEmbeds.length;
		final int numSeq = inputEmbeds[0].length;
		final float[][][] layernormOutput = new float[batchSize][numSeq][];
		for (int idxI = 0; idxI < batchSize; idxI++) {
			final int i = idxI;
			executor.startLoopTasks(numSeq, (jStart, jEnd) -> () -> {
				for (int j = jStart; j < jEnd; j++) {
					layernormOutput[i][j] = inputLayerNorm.normalize(hiddenStates[i][j]);
				}
			});
		}
		if (LOG.isLoggable(Level.FINER)) {
			for (float[] row : layernormOutput[0]) {
				LOG.finer("Layernorm-output: " + Arrays.toString(Arrays.copyOfRange(row, 0, 3)));
			}
		}

		attention.forward(layernormOutput, fusedQkv, alibi, hiddenStates, attentionMask, attentionResidual);
		if (LOG.isLoggable(Level.FINER)) {
			for (int h = 0; h < 3; h++) {
				LOG.finer("attention.out " + h + ": " + Arrays.toString(Arrays.copyOfRange(attentionResidual[0][h], 0, 3)));
			}
		}

		for (int idxI = 0; idxI < batchSize; idxI++) {
			final int i = idxI;
			executor.startLoopTasks(numSeq, (jStart, jEnd) -> () -> {
				for (int j = jStart; j < jEnd; j++) {
					layernormOutput[i][j] = postAttentionLayerNorm.normalize(attentionResidual[i][j]);
				}
			});
		}
		if (LOG.isLoggable(Level.FINER)) {
			for (float[] row : layernormOutput[0]) {
				LOG.finer("attn/Layernorm-output: " + Arrays.toString(Arrays.copyOfRange(row, 0, 3)));
			}
		}

		mlp.forward(layernormOutput, attentionResidual, output);
		if (LOG.isLoggable(Level.FINE)) {
			for (int h = 0; h < 3; h++) {
				LOG.fine("mlp.out " + h + ": " + Arrays.toString(Arrays.copyOfRange(output[0][h], 0, 3)));
			}
		}
	}

}
