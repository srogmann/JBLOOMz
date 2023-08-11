package org.rogmann.llm;

import java.util.concurrent.Phaser;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class manages a pool of worker-threads using a phaser.
 */
public class LlmWorkerPoolPhaser implements LlmExecutor {
	/** logger */
	private static final Logger LOGGER = Logger.getLogger(LlmWorkerPoolPhaser.class.getName());

	private final int nThreads;
	private final LlmWorkerThread[] pool;
	private final Phaser phaserStart;
	private final Phaser phaserFinished;
	
	private final AtomicBoolean isFinished = new AtomicBoolean(false);
	
	private final Runnable[] tasks;
	
	private int nCalls = 0;
	private int nCallsLoop = 0;

	public LlmWorkerPoolPhaser(final int nThreads) {
		this.nThreads = nThreads;
		LOGGER.info("Start thread-pool of " + nThreads + ((nThreads == 1) ? " thread" : " threads"));
		phaserStart = new Phaser(1 + nThreads) {
		     protected boolean onAdvance(int phase, int parties) { return false; }
		     };
		phaserFinished = new Phaser(1 + nThreads) {
		     protected boolean onAdvance(int phase, int parties) { return false; }
		     };
		pool = new LlmWorkerThread[1 + nThreads];
		for (int i = 0; i < nThreads; i++) {
			pool[i] = new LlmWorkerThread(i, phaserStart, phaserFinished);
			pool[i].start();
		}
		tasks = new Runnable[nThreads];
	}
	
	public void startTasks(LlmTaskFunction taskFunction) {
		for (int i = 0; i < nThreads; i++) {
			pool[i].task.set(taskFunction.apply(i, nThreads));
		}
		LOGGER.finer("Start Tasks");
		phaserStart.arrive();
		nCalls++;
		
		LOGGER.finer("Wait for tasks");
		phaserFinished.arriveAndAwaitAdvance();
		LOGGER.finer("End of Tasks");
	}

	@Override
	public void startLoopTasks(int n, LlmTaskLoopFunction taskFunction) {
		int blockSize = Math.max(1, n / nThreads);
		int idxStart = 0;
		int idxEnd = blockSize;
		for (int i = 0; i < nThreads; i++) {
			if (i == nThreads - 1 && idxEnd < n) {
				idxEnd = n;
			}
			pool[i].task.set(taskFunction.apply(idxStart, idxEnd));
			idxStart = Math.min(idxStart + blockSize, n);
			idxEnd =+ Math.min(idxEnd + blockSize, n);
		}
		LOGGER.finer("Start loop-tasks");
		phaserStart.arrive();
		nCallsLoop++;
		
		LOGGER.finer("Wait for loop-tasks");
		phaserFinished.arriveAndAwaitAdvance();
		LOGGER.finer("End of Tasks");
	}

	@Override
	public void close() {
		isFinished.set(true);
		LOGGER.info("close");
		for (LlmWorkerThread thread : pool) {
			if (thread != null) {
				thread.phaserStart.forceTermination();
			}
		}
		if (nCalls > 0) {
			LOGGER.info("nCalls: " + nCalls);
		}
		if (nCallsLoop > 0) {
			LOGGER.info("nCallsLoop: " + nCallsLoop);
		}
	}

	static class LlmWorkerThread extends Thread {
		/** logger */
		private static final Logger LOGGER = Logger.getLogger(LlmWorkerPoolPhaser.class.getName());

		private final int idxThread;
		private final Phaser phaserStart;
		private final Phaser phaserTaskFinished;
		final AtomicReference<Runnable> task = new AtomicReference<>();

		LlmWorkerThread(int idxThread, Phaser phaserStart, Phaser phaserTaskFinished) {
			super("LlmWorker-" + idxThread);
			this.idxThread = idxThread;
			this.phaserStart = phaserStart;
			this.phaserTaskFinished = phaserTaskFinished;
		}

		@Override
		public void run() {
			long cnt = 0;
			while (!phaserStart.isTerminated()) {
				phaserStart.arriveAndAwaitAdvance();
				if (phaserStart.isTerminated()) {
					break;
				}
				if (LOGGER.isLoggable(Level.FINER)) {
					LOGGER.finer("Thread " + idxThread + ": start runnable");
				}
				final Runnable runnable = task.get();
				runnable.run();
				phaserTaskFinished.arrive();
				cnt++;
			}
			if (LOGGER.isLoggable(Level.FINER)) {
				LOGGER.finer("PhaserStart: " + phaserStart);
				LOGGER.finer("PhaserTaskFinisheds: " + phaserTaskFinished);
				LOGGER.finer("Thread " + idxThread + ": terminates (count " + cnt + ")");
			}
		}
	}

}
