package org.rogmann.llm.bloom;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llm.LlmExecutor;
import org.rogmann.llm.nn.Tensor;

/**
 * Class used to compute an ALiBi-tensor.
 */
public class BloomAlibi {
	/** Logger */
	private static final Logger LOG = Logger.getLogger(BloomAlibi.class.getName());

	/**
	 * Computes a ALiBi-tensor (see https://arxiv.org/abs/2108.12409).
	 * 
	 * @param attentionMask attention-mask of shape (batchSize, maxSeqLen)
	 * @param numHeads number of heads
	 * @param executor executor
	 * @return 3d-tensor of shape (batchSize * numHeads, 1, maxSeqLen)
	 */
	public static Tensor buildAlibiTensor(final float[][] attentionMask,
			int numHeads, LlmExecutor executor) {
		final int batchSize = attentionMask.length;
		int closestPowerOf2 = 1;
		while (true) {
			int next = closestPowerOf2 * 2;
			if (next > numHeads) {
				break;
			}
			closestPowerOf2 = next;
		}
		float f1 = (float) Math.pow(2, -8.0f / closestPowerOf2);
		if (LOG.isLoggable(Level.FINE)) {
			LOG.fine("closestPowerOf2: " + closestPowerOf2);
			LOG.fine("d: " + f1);
		}
		
		final float[] slopes = new float[numHeads];
		float b = f1;
		for(int i = 0; i < closestPowerOf2; i++) {
			slopes[i] = b;
			b *= f1;
		}
		
		if (closestPowerOf2 < numHeads) {
			float f2 = (float) Math.pow(2, -4.0f / closestPowerOf2);
			b = f2;
			final float q = f2 * f2;
			for(int i=closestPowerOf2; i<numHeads; i++) {
				slopes[i] = b;
				b *= q;
			}
		}
		
		if (LOG.isLoggable(Level.FINE)) {
			LOG.fine("base = " + Arrays.toString(slopes));
		}
		
		final int maxSeqLen = attentionMask[0].length;
		final float[][] arange = new float[batchSize][maxSeqLen];
		for (int i = 0; i < batchSize; i++) {
			float sum = 0;
			for (int j = 0; j < maxSeqLen; j++) {
				sum += attentionMask[i][j];
				arange[i][j] = (sum - 1) * attentionMask[i][j];
			}
		}
		
		final Tensor tensor = new Tensor(batchSize * numHeads, 1, maxSeqLen, executor);
		float[][][] alibi = tensor.t3;
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < numHeads; j++) {
				for (int k = 0; k < maxSeqLen; k++) {
					alibi[i * numHeads + j][0][k] = arange[i][k] * slopes[j];
				}
			}
		}
		return tensor;
	}
}
