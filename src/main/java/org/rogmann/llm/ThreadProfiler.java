package org.rogmann.llm;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Very simple profiler.
 */
public class ThreadProfiler {
	private final Map<String, Long> mapMethodCount = new HashMap<>(1000);
	private final Map<String, String> mapMethodSampleTrace = new HashMap<>(1000);
	private final Map<String, Integer> mapMethodLine = new HashMap<>(1000);
	
	private final AtomicBoolean isRunning = new AtomicBoolean(false);

	private final Thread[] threads;
	private final long maxTicks;
	private final int sleep;
	private final int numMethodsToPrint;
	private final int maxStackDepth;
	
	private final ThreadWatcher watcher;
	private final PrintStream psOut;

	/**
	 * Constructor
	 * @param threads list of thread to profile
	 * @param maxTicks maximum number of ticks to profile
	 * @param sleep optional sleep in milliseconds between two ticks (0 = no sleep)
	 * @param numMethodsToPrint number of methods to display in the report
	 * @param maxStackDepth maximum number of methods in stack-trace
	 * @param psOut print stream to display the report
	 */
	public ThreadProfiler(Thread[] threads, long maxTicks, int sleep,
			int numMethodsToPrint, int maxStackDepth, PrintStream psOut) {
		this.threads = threads;
		this.maxTicks = maxTicks;
		this.sleep = sleep;
		this.numMethodsToPrint = numMethodsToPrint;
		this.maxStackDepth = maxStackDepth;
		this.psOut = psOut;
		
		watcher = new ThreadWatcher();
	}

	public void start() {
		isRunning.set(true);
		new Thread(watcher, "LlmProfiler").start();
	}

	public void stop() {
		isRunning.set(false);
	}

	class ThreadWatcher implements Runnable {

		@Override
		public void run() {
			long numTicks = 0;
			while (isRunning.get() && numTicks < maxTicks) {
				if (sleep > 0) {
					try {
						Thread.sleep(sleep);
					} catch (InterruptedException e) {
						Thread.currentThread().interrupt();
						break;
					}
				}
				for (Thread thread : threads) {
					final StackTraceElement[] aSte = thread.getStackTrace();
					if (aSte.length == 0) {
						continue;
					}
					final StackTraceElement ste = aSte[0];
					String cn = ste.getClassName();
					String mn = ste.getMethodName();
					final String key = new StringBuilder(cn.length() + 1 + mn.length())
							.append(cn).append('#').append(mn).toString();
					mapMethodCount.compute(key, (k, ticksPrev) ->
						Long.valueOf((ticksPrev == null) ? 1 : ticksPrev.longValue() + 1));
					mapMethodLine.put(key, Integer.valueOf(ste.getLineNumber()));
					if (mapMethodSampleTrace.get(key) == null) {
						final StringBuilder sb = new StringBuilder(100);
						for (int i = 1; i < aSte.length && i < maxStackDepth; i++) {
							final StackTraceElement lSte = aSte[i];
							if (i > 1) {
								sb.append(", ");
							}
							sb.append(lSte.getClassName()).append('#').append(lSte.getMethodName());
							sb.append('(').append(lSte.getLineNumber()).append(')');
						}
						mapMethodSampleTrace.put(key, sb.toString());
					}
				}
			}
			psOut.println("Profile finished");
			psOut.println("numTicks = " + numTicks);
			psOut.println("numMethods = " + mapMethodCount.size());
			mapMethodCount.entrySet().stream()
				.sorted((e1, e2) -> Long.compare(e2.getValue().longValue(), e1.getValue().longValue()))
				.limit(numMethodsToPrint)
				.forEach(entry -> {
					String method = entry.getKey();
					Integer line = mapMethodLine.get(method);
					String sampleTrace = mapMethodSampleTrace.get(method);
					psOut.println(String.format(" * %s (%d ticks, last line %d, %s)",
							method, entry.getValue(), line, sampleTrace));
				});
		}
	}
}
