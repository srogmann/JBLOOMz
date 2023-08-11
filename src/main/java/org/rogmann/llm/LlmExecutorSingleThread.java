package org.rogmann.llm;

import java.util.logging.Logger;

/**
 * Executor to use the current thread for computation.
 */
public class LlmExecutorSingleThread implements LlmExecutor {
	/** logger */
	private static final Logger LOGGER = Logger.getLogger(LlmExecutorSingleThread.class.getName());

	/** {@inheritDoc} */
	@Override
	public void startTasks(LlmTaskFunction taskFunction) {
		final Runnable runnable = taskFunction.apply(0, 1);
		runnable.run();
	}

	/** {@inheritDoc} */
	@Override
	public void startLoopTasks(int n, LlmTaskLoopFunction taskFunction) {
		final Runnable runnable = taskFunction.apply(0, n);
		runnable.run();
	}

	/** {@inheritDoc} */
	@Override
	public void close() {
		LOGGER.info("closed single-thread executor");
	}

}
