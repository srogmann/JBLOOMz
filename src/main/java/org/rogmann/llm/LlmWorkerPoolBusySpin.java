package org.rogmann.llm;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class manages a pool of worker-threads
 * using a CPU-intensive busy-spin.
 */
public class LlmWorkerPoolBusySpin implements LlmExecutor {
	/** logger */
	static final Logger LOGGER = Logger.getLogger(LlmWorkerPoolBusySpin.class.getName());

	private final int nThreads;
	private final LlmWorkerThread[] pool;
	
	private final AtomicBoolean isFinished = new AtomicBoolean(false);
	
	private final Runnable[] tasks;
	
	private int nCalls = 0;
	private int nCallsLoop = 0;

	public LlmWorkerPoolBusySpin(final int nThreads) {
		this.nThreads = nThreads;
		LOGGER.info("Start thread-pool of " + nThreads + ((nThreads == 1) ? " thread" : " threads"));
		pool = new LlmWorkerThread[1 + nThreads];
		for (int i = 0; i < nThreads; i++) {
			pool[i] = new LlmWorkerThread(i);
			pool[i].start();
		}
		tasks = new Runnable[nThreads];
	}

	public void startTasks(LlmTaskFunction taskFunction) {
		for (int i = 0; i < nThreads; i++) {
			pool[i].task.set(taskFunction.apply(i, nThreads));
			pool[i].lockStart.set(true);
		}
		LOGGER.finer("Start Tasks");
		nCalls++;
		
		LOGGER.finer("Wait for tasks");
		for (int i = 0; i < nThreads; i++) {
			while (!pool[i].lockFinished.compareAndSet(true, false));
		}
		LOGGER.finer("End of Tasks");
	}

	@Override
	public void startLoopTasks(int n, LlmTaskLoopFunction taskFunction) {
		int blockSize = Math.max(1, n / nThreads);
		int idxStart = 0;
		int idxEnd = blockSize;
		LOGGER.finer("Start loop-tasks");
		for (int i = 0; i < nThreads; i++) {
			if (i == nThreads - 1 && idxEnd < n) {
				idxEnd = n;
			}
			pool[i].task.set(taskFunction.apply(idxStart, idxEnd));
			pool[i].lockStart.set(true);
			idxStart = Math.min(idxStart + blockSize, n);
			idxEnd =+ Math.min(idxEnd + blockSize, n);
		}
		nCallsLoop++;
		LOGGER.finer("Wait for loop-tasks");
		for (int i = 0; i < nThreads; i++) {
			while (!pool[i].lockFinished.compareAndSet(true, false));
		}
		LOGGER.finer("End of Tasks");
	}

	@Override
	public void close() {
		isFinished.set(true);
		LOGGER.info("close");
		for (LlmWorkerThread thread : pool) {
			if (thread != null) {
				thread.lockStart.set(true);
			}
		}
		if (nCalls > 0) {
			LOGGER.info("nCalls: " + nCalls);
		}
		if (nCallsLoop > 0) {
			LOGGER.info("nCallsLoop: " + nCallsLoop);
		}
	}

	class LlmWorkerThread extends Thread {

		private final int idxThread;
		final AtomicBoolean lockStart = new AtomicBoolean(false);
		final AtomicBoolean lockFinished = new AtomicBoolean(false);
		final AtomicReference<Runnable> task = new AtomicReference<>();

		LlmWorkerThread(int idxThread) {
			super("LlmWorker-" + idxThread);
			this.idxThread = idxThread;
		}

		@Override
		public void run() {
			long cnt = 0;
			while (!isFinished.get()) {
				while (!lockStart.compareAndSet(true, false));
				try {
					if (isFinished.get()) {
						break;
					}
					if (LOGGER.isLoggable(Level.FINER)) {
						LOGGER.finer("Thread " + idxThread + ": start runnable");
					}
					final Runnable runnable = task.get();
					runnable.run();
				}
				finally {
					lockFinished.set(true);
				}
				cnt++;
			}
			if (LOGGER.isLoggable(Level.FINER)) {
				LlmWorkerPoolBusySpin.LOGGER.finer("Thread " + idxThread + ": terminates (count " + cnt + ")");
			}
		}
	}

}
