package org.rogmann.llm.bloom;

import org.rogmann.llm.LlmExecutor;

/**
 * GELU (Gaussian Error Linear Unit) used by BLOOM.
 * Adapted from Megatron-DeepSpeed code.
 * 
 * <p>x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))</p>
 */
public class BloomGELU {
	/** executor */
	private final LlmExecutor executor;

	/**
	 * Constructor
	 * @param executor executor
	 */
	public BloomGELU(LlmExecutor executor) {
		this.executor = executor;
	}

	/**
	 * forward-computation.
	 * @param input input
	 * @param output output (may be equal to input)
	 */
	public void forward(float[][][] input, float[][][] output) {
		final int d1 = input.length;
		final int d2 = input[0].length; 
		final int d3 = input[0][0].length;
		for (int i = 0; i < d1; i++) {
			final float[][] m1 = input[i];
			final float[][] m2 = output[i];
			executor.startLoopTasks(d2, (jStart, jEnd) -> () -> {
				for (int j = jStart; j < jEnd; j++) {
					final float[] r1 = m1[j];
					final float[] r2 = m2[j];
					for (int k = 0; k < d3; k++) {
						final float x = r1[k];
						final float z = 1f + 0.044715f * x * x;
						final float y = 1.0f + (float) (Math.tanh(0.79788456f * x * z));
						r2[k] = x * 0.5f * y;
						if (Float.isNaN(r2[k])) {
							throw new IllegalStateException("Nan");
						}
					}
				}
			});
		}
	}
}
