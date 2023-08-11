package org.rogmann.llm.bloom;

import org.rogmann.llm.LlmExecutor;
import org.rogmann.llm.nn.Linear;
import org.rogmann.llm.nn.Tensor;

/**
 * MLP (multilayer perceptron) of BLOOM.
 */
public class BloomMLP {

	private final Linear denseHTo4H;
	private final Linear dense4HToH;
	private final BloomGELU gelu;

	public BloomMLP(final Linear denseHTo4H, final Linear dense4HToH, LlmExecutor executor) {
		this.denseHTo4H = denseHTo4H;
		this.dense4HToH = dense4HToH;
		this.gelu = new BloomGELU(executor);
	}
	
	/**
	 * Does a feedforward computation.
	 * @param hiddenStates input-tensor (batchSize, numSeq, hiddenSize)
	 * @param residual residual-tensor (batchSize, numSeq, hiddenSize)
	 * @param output output-tensor (batchSize, numSeq, hiddenSize)
	 */
	public void forward(float[][][] hiddenStates, float[][][] residual, final float[][][] output) {
		final int batchSize = hiddenStates.length;
		final int numSeq = hiddenStates[0].length;
		final int dimHidden = hiddenStates[0][0].length;

		final float[][][] hidden4H = new float[batchSize][numSeq][4 * dimHidden];
		denseHTo4H.mult(hiddenStates, hidden4H);
		gelu.forward(hidden4H, hidden4H);

		dense4HToH.mult(hidden4H, output);

		Tensor.add(output, residual, output);
	}
}
