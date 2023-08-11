package org.rogmann.llm;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class manages a pool of worker-threads.
 */
public class LlmWorkerPoolReentrantLock implements LlmExecutor {
	/** logger */
	static final Logger LOGGER = Logger.getLogger(LlmWorkerPoolReentrantLock.class.getName());

	private final int nThreads;
	private final LlmWorkerThread[] pool;
	
	private final AtomicBoolean isFinished = new AtomicBoolean(false);
	
	private final Runnable[] tasks;
	
	private int nCalls = 0;
	private int nCallsLoop = 0;

	public LlmWorkerPoolReentrantLock(final int nThreads) {
		this.nThreads = nThreads;
		LOGGER.info("Start thread-pool of " + nThreads + ((nThreads == 1) ? " thread" : " threads"));
		pool = new LlmWorkerThread[1 + nThreads];
		for (int i = 0; i < nThreads; i++) {
			pool[i] = new LlmWorkerThread(i);
			pool[i].lockStart.lock();
			pool[i].start();
		}
		tasks = new Runnable[nThreads];
	}

	public void startTasks(LlmTaskFunction taskFunction) {
		for (int i = 0; i < nThreads; i++) {
			pool[i].task.set(taskFunction.apply(i, nThreads));
			pool[i].lockFinished.lock();
			pool[i].lockStart.unlock();
		}
		LOGGER.finer("Start Tasks");
		nCalls++;
		
		LOGGER.finer("Wait for tasks");
		for (int i = 0; i < nThreads; i++) {
			pool[i].lockStart.lock();
		}
		for (int i = 0; i < nThreads; i++) {
			pool[i].lockFinished.unlock();
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
			pool[i].lockFinished.lock();
			pool[i].lockStart.unlock();
			idxStart = Math.min(idxStart + blockSize, n);
			idxEnd =+ Math.min(idxEnd + blockSize, n);
		}
		nCallsLoop++;
		LOGGER.finer("Wait for loop-tasks");
		for (int i = 0; i < nThreads; i++) {
			pool[i].lockStart.lock();
		}
		for (int i = 0; i < nThreads; i++) {
			pool[i].lockFinished.unlock();
		}
		LOGGER.finer("End of Tasks");
	}

	@Override
	public void close() {
		isFinished.set(true);
		LOGGER.info("close");
		for (LlmWorkerThread thread : pool) {
			if (thread != null) {
				thread.lockStart.unlock();
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
		final Lock lockStart = new ReentrantLock(true);
		final Lock lockFinished = new ReentrantLock(true);
		final AtomicReference<Runnable> task = new AtomicReference<>();

		LlmWorkerThread(int idxThread) {
			super("LlmWorker-" + idxThread);
			this.idxThread = idxThread;
		}

		@Override
		public void run() {
			long cnt = 0;
			while (!isFinished.get()) {
				lockStart.lock();
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
					lockStart.unlock();
					lockFinished.lock();
					lockFinished.unlock();
				}
				cnt++;
			}
			if (LOGGER.isLoggable(Level.FINER)) {
				LlmWorkerPoolReentrantLock.LOGGER.finer("Thread " + idxThread + ": terminates (count " + cnt + ")");
			}
		}
	}

}
