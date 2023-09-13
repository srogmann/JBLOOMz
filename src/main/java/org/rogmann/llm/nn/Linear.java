package org.rogmann.llm.nn;

import org.rogmann.llm.LlmExecutor;

/**
 * Class concerning matrix multiplications.
 */
public class Linear {

	private final int dim1;
	private final int dim2;
	private float[][] mat;
	private float[] bias;
	private final LlmExecutor executor;

	/**
	 * Constructor
	 * @param mat matrix
	 * @param executor executor
	 */
	public Linear(float[][] mat, LlmExecutor executor) {
		this.dim1 = mat.length;
		this.dim2 = mat[0].length;
		this.mat = mat;
		this.bias = new float[dim1];
		this.executor = executor;
	}
	
	/**
	 * Constructor
	 * @param mat matrix
	 * @param bias bias
	 * @param executor executor
	 */
	public Linear(float[][] mat, float[] bias, LlmExecutor executor) {
		this.dim1 = mat.length;
		this.dim2 = mat[0].length;
		this.mat = mat;
		this.bias = bias;
		this.executor = executor;
	}

	/**
	 * Computes input * transposed(mat) + bias.
	 * @param input input
	 * @param output result
	 */
	public void mult(float[][][] input, final float[][][] output) {
		final int dimBatch = input.length;
		final int d = input[0].length;
		if (input[0][0].length != dim2 || d > output[0].length || output[0][0].length != dim1) {
			throw new IllegalArgumentException(String.format("mult: dimension mismatch, input (%d, %d, %d), mat (%d, %d), output(%d, %d, %d)",
					input.length, input[0].length, input[0][0].length,
					mat.length, mat[0].length,
					output.length, output[0].length, output[0][0].length));
		}
		if (d == 1) {
			for (int idxB = 0; idxB < dimBatch; idxB++) {
				final int b = idxB;
				final int i = 0;
				executor.startLoopTasks(dim1, (jStart, jEnd) -> () -> {
					for (int j = jStart; j < jEnd; j++) {
						float sum = bias[j];
						for (int k = 0; k < dim2; k++) {
							sum += input[b][i][k] * mat[j][k];
						}
						output[b][i][j] = sum;
					}
				});
			}
		}
		else {
			for (int idxB = 0; idxB < dimBatch; idxB++) {
				final int b = idxB;
				executor.startLoopTasks(d, (iStart, iEnd) -> () -> {
					for (int i = iStart; i < iEnd; i++) {
						for (int j = 0; j < dim1; j++) {
							float sum = bias[j];
							for (int k = 0; k < dim2; k++) {
								sum += input[b][i][k] * mat[j][k];
							}
							output[b][i][j] = sum;
						}
					}
				});
			}
		}
	}

	/**
	 * Computes input * transposed(mat) + bias.
	 * This method can be used when a part of the result has already been computed.
	 * @param input input
	 * @param output result
	 * @param startOffsetOutputDim2 start-offset in the second dimension in the output-tensor
	 */
	public void multMinDim2(float[][][] input, final float[][][] output, final int startOffsetOutputDim2) {
		final int dimBatch = input.length;
		final int d = input[0].length;
		if (input[0][0].length != dim2 || d > output[0].length || output[0][0].length != dim1) {
			throw new IllegalArgumentException(String.format("mult: dimension mismatch, input (%d, %d, %d), mat (%d, %d), output(%d, %d, %d)",
					input.length, input[0].length, input[0][0].length,
					mat.length, mat[0].length,
					output.length, output[0].length, output[0][0].length));
		}
		for (int idxB = 0; idxB < dimBatch; idxB++) {
			final int b = idxB;
			for (int idxI = 0; idxI < d; idxI++) {
				final int i = idxI;
				executor.startLoopTasks(dim1, (jStart, jEnd) -> () -> {
					for (int j = jStart; j < jEnd; j++) {
						float sum = bias[j];
						for (int k = 0; k < dim2; k++) {
							sum += input[b][i][k] * mat[j][k];
						}
						output[b][startOffsetOutputDim2 + i][j] = sum;
					}
				});
			}
		}
	}

}
