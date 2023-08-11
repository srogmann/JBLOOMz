package org.rogmann.llm;

/**
 * Interface to create a runnable to be executed in a worker-thread.
 */
@FunctionalInterface
public interface LlmTaskFunction {

	/**
	 * Creates a runnable to executed a part of work in a worker-thread.
	 * @param idxTask index of the worker-thread
	 * @param nTasks number of worker-threads.
	 * @return runnable to be executed in the worker-thread of given index
	 */
	Runnable apply(int idxTask, int nTasks);

}
