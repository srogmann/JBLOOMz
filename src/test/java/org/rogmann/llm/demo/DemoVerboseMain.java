package org.rogmann.llm.demo;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import org.rogmann.llm.LlmConfigException;
import org.rogmann.llm.LlmExecutor;
import org.rogmann.llm.LlmWorkerPoolPhaser;
import org.rogmann.llm.ModelReader;
import org.rogmann.llm.ThreadProfiler;
import org.rogmann.llm.bloom.BloomModel;
import org.rogmann.llm.tokenizer.BPETokenizer;
import org.rogmann.llm.tokenizer.Tokenizer;

/**
 * Executes a text generation using a BLOOM based model.
 */
public class DemoVerboseMain {
	/** logger */
	private static final Logger LOG = Logger.getLogger(DemoVerboseMain.class.getName());

	/**
	 * Entry method.
	 * @param args model-folder
	 */
	public static void main(String[] args) throws IOException, LlmConfigException {
		if (args.length == 0) {
			throw new IllegalArgumentException("Usage: model-folder");
		}
		final File folder = new File(args[0]);
		final Tokenizer tokenizer = new BPETokenizer(folder);
		
		System.out.println("#tokens in tokenizer: " + tokenizer.size());
		final Instant tsStartLoad = Instant.now();

		final boolean supportUnzippedModel = true;
		final ModelReader modelReader = new ModelReader(folder, supportUnzippedModel);
		final int nThreads = 8;
		final boolean useProfiler = Boolean.FALSE.booleanValue();

		final Instant tsStartInfer;
		ThreadProfiler profiler = null;
		if (useProfiler) {
			profiler = new ThreadProfiler(new Thread[] { Thread.currentThread() }, Long.MAX_VALUE, 0, 20, 5, System.out);
		}

		try (LlmExecutor executor = new LlmWorkerPoolPhaser(nThreads)) {
		//try (LlmExecutor executor = new LlmWorkerPoolReentrantLock(nThreads)) {
		//try (LlmExecutor executor = new LlmWorkerPoolBusySpin(nThreads)) {
		//try (LlmExecutor executor = new LlmExecutorSingleThread()) {
	
			final int maxBatchSize = 3;
			final BloomModel model = new BloomModel(modelReader, maxBatchSize, executor);
	
			tsStartInfer = Instant.now();
			//String inputSentence = "Auf der Wiese läuft ein Hund hinter";
			//String inputSentence = "Der Hund heißt Karl. Die Katze heißt Mimi. Wie nennt Mimi den Hund?";
			String inputSentence = "Translate to Chinese: I write a program in Java.";
			//String inputSentence = "What is the capital of France?";
			//String inputSentence = "Translate to chinese: cat.";
			//String inputSentence = "¿Quién era Joan Miró?";
			int[][] inputIds = tokenizer.encode(inputSentence);
			final int maxToken = 10;
			if (profiler != null) {
				profiler.start();
			}
			final List<String> listToken = new ArrayList<>();
			for(int idxInf = 1; idxInf <= maxToken; idxInf++) {
				System.out.println("");
				System.out.println("Inference " + idxInf);
				System.out.println("input_ids: " + Arrays.toString(inputIds[0]));
				System.out.println("Start: " + LocalDateTime.now());
				
				float[][][] hiddenState = model.forward(inputIds);

				final float[] lastState = hiddenState[0][hiddenState[0].length - 1];
				if (LOG.isLoggable(Level.FINE)) {
					LOG.fine("Last State: " + Arrays.toString(Arrays.copyOfRange(lastState, 0, 3)));
					LOG.fine("     [...]  " + Arrays.toString(Arrays.copyOfRange(lastState, lastState.length - 3, lastState.length)));
					LOG.fine("Tok.size: " + tokenizer.size());
					LOG.fine("lastState.len: " + lastState.length);
				}
				final float[] lastEmbedding = model.getEmbeddings().computeLastEmbedding(lastState);
				int idx = 0;
				float max = -1e10f;
				for (int i = 0; i < lastEmbedding.length; i++) {
					if (lastEmbedding[i] > max) {
						max = lastEmbedding[i];
						idx = i;
						String token = tokenizer.decode(i);
						System.out.println(String.format("idx=%d, max=%f, token=%s (%s)",
								idx, max, token, tokenizer.convertToInternal(token)));
					}
				}
				String sToken = tokenizer.decode(idx);
				System.out.println("End: " + LocalDateTime.now());
				System.out.println("Token: " + sToken);
				listToken.add(sToken);
				
				inputIds = tokenizer.appendToken(inputIds, idx);
			}
			System.out.println("Prompt: " + inputSentence);
			final String sResult = listToken.stream().collect(Collectors.joining());
			System.out.println("Result:" + sResult);
		}
		
		final Instant tsEnd = Instant.now();
		System.out.println("nThreads: " + nThreads);
		System.out.println("Duration Load: " + Duration.between(tsStartLoad, tsStartInfer));
		System.out.println("Duration Infer: " + Duration.between(tsStartInfer, tsEnd));
		if (profiler != null) {
			profiler.stop();
		}

	}
}
