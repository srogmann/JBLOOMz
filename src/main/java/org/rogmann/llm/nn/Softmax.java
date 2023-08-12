package org.rogmann.llm.nn;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

import org.rogmann.llm.LlmExecutor;

/**
 * Compute the softmax function (normalized exponential function).
 */
public class Softmax {
	/** logger */
	private static final Logger LOG = Logger.getLogger(Softmax.class.getName());
	/** <code>true</code> if a substraction of a value is necessary */
	private static final AtomicBoolean SWITCHED_TO_SOFTMAX_MINUS_MAX = new AtomicBoolean(false);
	

	/**
	 * Computes the softmax of a vector.
	 * @param input input
	 * @return output
	 */
	public static float[] softmax(float[] input) {
		final int dim = input.length;
		float denom = 0;
		final float[] output = new float[dim];
		for (int i = 0; i < dim; i++) {
			float e = (float) Math.exp(input[i]);
			if (Float.isInfinite(e)) {
				throw new IllegalStateException(String.format("Infinity: input[%d]=%f, exp=%f, denom=%f",
						i, input[i], e, denom));
			}
			output[i] = e;
			denom += e;
		}
		for (int i = 0; i < dim; i++) {
			output[i] /= denom;
		}
		return output;
	}

	/**
	 * Computes the softmax of a 2d-tensor.
	 * @param input input
	 * @return output
	 */
	public static float[][] softmax(float[][] input) {
		final int d1 = input.length;
		float denom = 0;
		final float[][] output = new float[d1][];
		for (int i = 0; i < d1; i++) {
			final float[] rowIn = input[i];
			final int dim = rowIn.length;
			final float[] rowOut = new float[dim];
			for (int j = 0; j < dim; j++) {
				float e = (float) Math.exp(rowIn[j]);
				rowOut[j] = e;
				denom += e;
			}
			output[i] = rowOut;
		}
		for (int i = 0; i < d1; i++) {
			final float[] rowOut = output[i];
			final int dim = rowOut.length;
			for (int j = 0; j < dim; j++) {
				rowOut[i] /= denom;
			}
		}
		return output;
	}

	/**
	 * Computes the softmax of a 4d-tensor in-place in the last dimension.
	 * @param input input and output
	 * @param executor executor
	 */
	public static void softmaxInlineLastDim(float[][][][] input, LlmExecutor executor) {
		for (float[][][] mat1 : input) {
			final int d1 = mat1.length;
			executor.startLoopTasks(d1, (hStart, hEnd) -> () -> {
				final double[] tmp = new double[input[0][0].length];
				for (int h = hStart; h < hEnd; h++) {
					float[][] mat2 = mat1[h];
					for (float[] row : mat2) {
						final int d = row.length;
						float max = 0;
						for (float r : row) {
							if (r > max) {
								max = r;
							}
						}
						// maximum of double is exp(709.78), we want to avoid infinity.
						if (max < 20) {
							double denom = 0;
							for (int j = 0; j < d; j++) {
								double r = row[j];
								double e = Math.exp(r);
								tmp[j] = e;
								denom += e;
							}
							for (int j = 0; j < d; j++) {
								final double t = tmp[j];
								row[j] = (t > 0) ? (float) (tmp[j] / denom) : 0f;
								if (Float.isNaN(row[j])) {
									throw new IllegalStateException(String.format("NaN: %f = %f / %f, len=%d",
											row[j], tmp[j], denom, row.length));
								}
							}
						}
						else {
							if (!SWITCHED_TO_SOFTMAX_MINUS_MAX.getAndSet(true)) {
								LOG.info(String.format("Switched to softmax minus max: h=%d, max=%.1f", h, max));
							}
							double denom = 0;
							for (int j = 0; j < d; j++) {
								double r = row[j] - max;
								double e = Math.exp(r);
								tmp[j] = e;
								denom += e;
							}
							for (int j = 0; j < d; j++) {
								row[j] = (float) (tmp[j] / denom);
							}
						}
					}
				}
			});
		}
	}

}
