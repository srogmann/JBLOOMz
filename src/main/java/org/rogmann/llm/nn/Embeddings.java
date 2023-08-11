package org.rogmann.llm.nn;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llm.LlmExecutor;

/**
 * Class used to embed tokens in hidden states.
 */
public class Embeddings {
	/** logger */
	private static final Logger LOG = Logger.getLogger(Embeddings.class.getName());

	/** weights */
	private float[][] weights;
	/** executor */
	private LlmExecutor executor;

	/**
	 * Constructor
	 * @param tWeights weights
	 * @param executor executor
	 */
	public Embeddings(float[][] tWeights, LlmExecutor executor) {
		this.weights = tWeights;
		this.executor = executor;
	}

	/**
	 * Embeds tokens into hidden state vectors.
	 * @param inputIds batch of input-tokens
	 * @return embedded tokens (batchSize, inputSize, dim of weights)
	 */
	public float[][][] wordEmbeddings(int[][] inputIds) {
		final int dim = weights[0].length;
		final int batchSize = inputIds.length;
		final int inputSize = inputIds[0].length;
		final float[][][] output = new float[batchSize][inputSize][dim];
		for (int i = 0; i < batchSize; i++) {
			final int[] input = inputIds[i];
			for (int j = 0; j < inputSize; j++) {
				final int token = input[j];
				final float[] hiddenVector = output[i][j];
				System.arraycopy(weights[token], 0, hiddenVector, 0, dim);
			}
		}
		return output;
	}

	public float[] computeLastEmbedding(final float[] lastState) {
		final float[] lastEmbedding = new float[weights.length];
		executor.startLoopTasks(weights.length, (iStart, iEnd) -> () -> {
			for (int i = iStart; i < iEnd; i++) {
				float sum = 0;
				for (int j = 0; j < lastState.length; j++) {
					sum += weights[i][j] * lastState[j];
				}
				lastEmbedding[i] = sum;
			}
		});
		if (LOG.isLoggable(Level.FINE)) {
			LOG.fine("Last Embed: " + Arrays.toString(Arrays.copyOfRange(lastEmbedding, 0, 3)));
		}
		return lastEmbedding;
	}

}
