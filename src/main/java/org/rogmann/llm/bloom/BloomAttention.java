package org.rogmann.llm.bloom;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llm.LlmExecutor;
import org.rogmann.llm.nn.Linear;
import org.rogmann.llm.nn.Softmax;
import org.rogmann.llm.nn.Tensor;

/**
 * Class to compute the scaled dot-product of the attention in the BLOOM-model.
 * This includes the addition of the ALiBi-tensor.
 */
public class BloomAttention {
	/** Logger */
	private static final Logger LOG = Logger.getLogger(BloomAttention.class.getName());

	private final int numHeads;
	private final int headDim;

	private final float invNormFactor;
	private final float beta;
	/**
	 * dim1 = 3 * hiddenSize, dim2 = hiddenSize.
	 * hiddenSize -&gt; 3 * hiddenSize: matrix used to build query, key and value. */
	private final Linear queryKeyValue;
	private final Linear dense;

	private final LlmExecutor executor;
	

	/**
	 * Constructor
	 * @param hiddenSize size of hidden layer
	 * @param numHeads number of attention-head
	 * @param queryKeyValue weights to compute query, key and value
	 * @param dense linear dense transformation
	 * @param executor executor
	 */
	public BloomAttention(final int hiddenSize, final int numHeads, Linear queryKeyValue, Linear dense,
			LlmExecutor executor) {
		this.numHeads = numHeads;
		headDim = hiddenSize / numHeads;
		
		invNormFactor = (float) (1.0 / Math.sqrt(headDim));
		beta = 1.0f;
		
		// queryKeyValue: hiddenSize -> 3 * hiddenSize
		this.queryKeyValue = queryKeyValue;
		// dense: hiddenSize -> hiddenSize
		this.dense = dense;
		
		this.executor = executor;
	}
	
	/**
	 * Computes an attention.
	 * 
	 * <p>
	 * @param hiddenStates input-tensor (batchSize, numSeq, hiddenSize)
	 * @param fusedQkv temporary tensor ([batchSize][numSeq][3 * dimHidden])
	 * @param numSeqLenCache <code>null</code> if no cache is used, numSeq in fusedQkv otherwise
	 * @param alibi ALiBi-tensor of shape executor(batchSize * numHeads, 1, numSeq)
	 * @param residual residual-tensor (batchSize, numSeq, hiddenSize)
	 * @param attentionMask attention-mask (batchSize, 1, maxSeqLen, maxSeqLen)
	 * @param output output-tensor (batchSize, seqLength, numHeads * headDim)
	 */
	public void forward(float[][][] hiddenStates,
			final float[][][] fusedQkv, final Integer numSeqLenCache,
			final Tensor alibi,
			final float[][][] residual, final boolean[][][][] attentionMask,
			final float[][][] output) {
		final int batchSize = hiddenStates.length;
		final int numSeq = hiddenStates[0].length;
		queryKeyValue.mult(hiddenStates, fusedQkv);
		if (LOG.isLoggable(Level.FINER) ) {
			LOG.finer(String.format("qKV (%d, %d, %d)", fusedQkv.length, fusedQkv[0].length, fusedQkv[0][0].length));
			for(float[] row : fusedQkv[0]) {
				LOG.finer("qKV: " + Arrays.toString(Arrays.copyOfRange(row, 0, 3)));
			}
		}
		// fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
		// -> 3 * dimHidden = numHeads * 3 * headDim

		// query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
		// key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
		// value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
		// -> queryLayer (batchSize * numHeads, numSeq, headDim)
		// -> keyLayer (batchSize * numHeads, headDim, numSeq)
		// -> valueLayer (batchSize * numHeads, numSeq, headDim)s
		// queryLayer[b + i * headDim][j][k] = fusedQkv[b][j][(i * 3 + 0) * headDim + k]
		// keyLayer[b + i * headDim][k][j] = fusedQkv[b][j][(i * 3 + 1) * headDim + k]
		// valueLayer[b + i * headDim][j][k] = fusedQkv[b][j][(i * 3 + 2) * headDim + k]

		float[][][][] multResult = new float[batchSize][numHeads][numSeq][numSeq];
		alibi.baddbmmView4(numSeq, fusedQkv, 3, numHeads, headDim, 0, 1,
				invNormFactor, beta, multResult);
		if (LOG.isLoggable(Level.FINER) ) {
			LOG.finer(String.format("multResult (%d, %d, %d, %d)",
					multResult.length, multResult[0].length, multResult[0][0].length, multResult[0][0][0].length));
			for (int h = 0; h < 3; h++) {
				for (int r = 0; r < 3; r++) {
					LOG.finer("Head " + h + ", row " + r + ": " + Arrays.toString(multResult[0][h][r]));
				}
			}
		}

		// multResult has to be viewed as (batchSize, numHeads, numSeq, numSeq)
		// -> attention_scores

		if (LOG.isLoggable(Level.FINER) ) {
			LOG.finer(String.format("attionmask.size (%d, %d)", attentionMask.length, attentionMask[0].length));
		}
		for (int b = 0; b < batchSize; b++) {
			for (int h = 0; h < numHeads; h++) {
				for (int i = 0; i < numSeq; i++) {
					for (int j = 0; j < numSeq; j++) {
						if (attentionMask[b][0][i][j]) {
							multResult[b][h][i][j] = -3.4028e+38f;
						}
					}
				}
			}
		}

		Softmax.softmaxInlineLastDim(multResult, executor);

		if (LOG.isLoggable(Level.FINER) ) {
			LOG.finer("After softmax");
			for (int h = 0; h < 3; h++) {
				for (int r = 0; r < 3; r++) {
					LOG.finer("Head " + h + ", row " + r + ": " + Arrays.toString(multResult[0][h][r]));
				}
			}
		}

		// No attention dropout.

		// fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
		// -> 3 * dimHidden = numHeads * 3 * headDim
		// value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
		// -> valueLayer (batchSize * numHeads, numSeq, headDim)s
		// valueLayer[b + i * headDim][j][k] = fusedQkv[b][j][(i * 3 + 2) * headDim + k]
		//
		//float[][][][] multResult = new float[batchSize][numHeads][numSeq][numSeq];
		final float[][][] contextLayer = new float[batchSize][numSeq][numHeads * headDim];
		Tensor.bmmView4(multResult, fusedQkv, numSeq, 3, numHeads, headDim, 2, contextLayer, executor);

		if (LOG.isLoggable(Level.FINER) ) {
			LOG.finer("after bmmView4");
			for (int h = 0; h < 3; h++) {
				LOG.finer("CtxLayer " + h + ": " + Arrays.toString(Arrays.copyOfRange(contextLayer[0][h], 0, 3)));
			}
		}

		dense.mult(contextLayer, output);

		Tensor.add(output, residual, output);
	}
}
