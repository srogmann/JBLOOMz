package org.rogmann.llm;

/**
 * Interface used to create a runnable to execute a part of a loop in a worker-thread.
 */
@FunctionalInterface
public interface LlmTaskLoopFunction {

	/**
	 * Create a runnable to execute a part of a loop.
	 * @param idxStart first index
	 * @param idxEnd last index (exclusive)
	 * @return runnable to be executed in a worker-thread
	 */
	Runnable apply(int idxStart, int idxEnd);

}
