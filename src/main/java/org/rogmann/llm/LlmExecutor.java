package org.rogmann.llm;

/**
 * Interface used to do a computation in a pool of worker-threads.
 */
public interface LlmExecutor extends AutoCloseable {

	/**
	 * Starts a task to be executed in parallel.
	 * @param taskFunction task-function
	 */
	void startTasks(LlmTaskFunction taskFunction);

	/**
	 * Starts a task consisting of a loop to be executed in parallel.
	 * @param n number of items to be processed (loop from 0 to n-1 inclusive)
	 * @param taskFunction task-loop-function
	 */
	void startLoopTasks(int n, LlmTaskLoopFunction taskFunction);

	/**
	 * Shutdown of the executor.
	 */
	void close();

}
