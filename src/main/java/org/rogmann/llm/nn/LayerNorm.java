package org.rogmann.llm.nn;

/**
 * Computes the layer norm.
 * 
 * <p>See also the paper "Layer Normalization" at https://arxiv.org/abs/1607.06450.</p>
 */
public class LayerNorm {

	private final float eps;
	/** gain respective weight */
	private final float[] alpha;
	/** bias */
	private final float[] beta;

	/**
	 * Constructor
	 * @param eps value added inside the square-root
	 * @param alpha gain respective weight
	 * @param beta bias
	 */
	public LayerNorm(float eps, float[] alpha, float[] beta) {
		this.eps = eps;
		this.alpha = alpha;
		this.beta = beta;
	}

	/**
	 * Computes the layer normalization of a 1d-tensor.
	 * The gain is set to 1, the bias is set to 0.
	 * @param input input
	 * @param eps value added to the denominator
	 * @return normalized input
	 */
	public static float[] normalize(float[] input, float eps) {
		final int d1 = input.length;

		float mu = 0;
		for (float a : input) {
			mu += a;
		}
		mu /= d1;

		float sigma = 0;
		for (float a : input) {
			float d = (a - mu);
			sigma += d * d;
		}
		sigma = (float) Math.sqrt(sigma / d1 + eps);

		float denom = sigma;
		final float[] output = new float[d1];
		for (int j = 0; j < d1; j++) {
			output[j] = (input[j] - mu) / denom;
		}
		
		return output;
	}

	/**
	 * Computes the layer normalization of a 1d-tensor.
	 * The gain is set to 1, the bias is set to 0.
	 * @param input input
	 * @return normalized input
	 */
	public float[] normalize(float[] input) {
		final int d1 = input.length;

		float mu = 0;
		for (float a : input) {
			mu += a;
		}
		mu /= d1;

		float sigma = 0;
		for (float a : input) {
			float d = (a - mu);
			sigma += d * d;
		}
		sigma = (float) Math.sqrt(sigma / d1 + eps);

		float denom = sigma;
		final float[] output = new float[d1];
		for (int j = 0; j < d1; j++) {
			output[j] = ((input[j] - mu) * alpha[j]) / denom + beta[j];
		}
		
		return output;
	}

	/**
	 * Computes the layer normalization of a 2d-tensor.
	 * The gain is set to 1, the bias is set to 0.
	 * @param input input
	 * @param eps value added to the denominator
	 * @return normalized input
	 */
	public static float[][] normalize(float[][] input, float eps) {
		final int d1 = input.length;
		final int d2 = input[0].length;

		float mu = 0;
		for (float[] row : input) {
			for (float a : row) {
				mu += a;
			}
		}
		mu /= (d1 * d2);

		float sigma = 0;
		for (float[] row : input) {
			for (float a : row) {
				float d = (a - mu);
				sigma += d * d;
			}
		}
		sigma = (float) Math.sqrt(sigma / (d1 * d2) + eps);

		float denom = sigma;
		final float[][] output = new float[d1][d2];
		for (int i = 0; i < d1; i++) {
			float[] row = input[i];
			for (int j = 0; j < d2; j++) {
				output[i][j] = (row[j] - mu) / denom;
			}
		}
		
		return output;
	}

	/**
	 * Computes the layer normalization of a 2d-tensor.
	 * @param input input
	 * @return normalized input
	 */
	public float[][] normalize(float[][] input) {
		final int d1 = input.length;
		final int d2 = input[0].length;

		float mu = 0;
		for (float[] row : input) {
			for (float a : row) {
				mu += a;
			}
		}
		mu /= (d1 * d2);

		float sigma = 0;
		for (float[] row : input) {
			for (float a : row) {
				float d = (a - mu);
				sigma += d * d;
			}
		}
		sigma = (float) Math.sqrt(sigma / (d1 * d2) + eps);

		final float denom = sigma;
		final float[][] output = new float[d1][d2];
		for (int idxI = 0; idxI < d1; idxI++) {
			final int i = idxI;
			float[] row = input[i];
			for (int j = 0; j < d2; j++) {
				output[i][j] = ((row[j] - mu) * alpha[j]) / denom + beta[j];
			}
		}
		
		return output;
	}

}
