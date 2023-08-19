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
			String inputSentence = "Der Hund heißt Karl. Die Katze heißt Mimi. Wie nennt Mimi den Hund?";
			//String inputSentence = "Translate to Chinese: I write a program in Java.";
			//String inputSentence = "What is the capital of France?";
			//String inputSentence = "Translate to chinese: cat.";
			//String inputSentence = "¿Quién era Joan Miró?";
			//String inputSentence = "Auf der Wiese steht eine Kuh unter ";
			int[][] inputIds = tokenizer.encode(inputSentence);
			final int maxToken = 10;
			final int numBeams = maxBatchSize;
			if (profiler != null) {
				profiler.start();
			}
			final List<List<String>> listBatchesToken = new ArrayList<>();
			listBatchesToken.add(new ArrayList<>());
			for(int idxInf = 1; idxInf <= maxToken; idxInf++) {
				System.out.println("");
				System.out.println("Inference " + idxInf);
				System.out.println("input_ids: " + Arrays.toString(inputIds[0]));
				System.out.println("Start: " + LocalDateTime.now());
				
				float[][][] hiddenState = model.forward(inputIds);

				int batchSize = hiddenState.length;
				final List<Integer> idxCandidates = new ArrayList<>();
				int[][] nextInputIds = new int[batchSize][];
				for (int batch = 0; batch < batchSize; batch++) {
					System.out.println("Batch " + batch);
					final float[][] batchState = hiddenState[batch];
					final float[] lastState = batchState[batchState.length - 1];
					if (LOG.isLoggable(Level.FINE)) {
						LOG.fine("Last State: " + Arrays.toString(Arrays.copyOfRange(lastState, 0, 3)));
						LOG.fine("     [...]  " + Arrays.toString(Arrays.copyOfRange(lastState, lastState.length - 3, lastState.length)));
						LOG.fine("Tok.size: " + tokenizer.size());
						LOG.fine("lastState.len: " + lastState.length);
					}
					final float[] lastEmbedding = model.getEmbeddings().computeLastEmbedding(lastState);
					int idx = 0;
					float max = -1e10f;
					idxCandidates.clear();
					for (int i = 0; i < lastEmbedding.length; i++) {
						if (lastEmbedding[i] > max) {
							max = lastEmbedding[i];
							idx = i;
							String token = tokenizer.decode(i);
							System.out.println(String.format(" idx=%d, max=%f, token=%s (%s)",
									idx, max, token, tokenizer.convertToInternal(token)));
							idxCandidates.add(Integer.valueOf(idx));
						}
					}
					String sToken = tokenizer.decode(idx);
					System.out.println(" End: " + LocalDateTime.now());
					System.out.println(" Token: " + sToken);
					final List<String> listToken = listBatchesToken.get(batch);
					listToken.add(sToken);
					nextInputIds[batch] = tokenizer.appendToken(inputIds[batch], idx);
				}

				final int numCandidates = idxCandidates.size();
				if (idxInf == 2 && numCandidates >= numBeams) {
					inputIds = new int[numBeams][];
					inputIds[0] = nextInputIds[0];
					for (int j = 1; j < numBeams; j++) {
						inputIds[j] = nextInputIds[0].clone();
						final int idx = idxCandidates.get(numCandidates - 1 - j).intValue();
						inputIds[j][inputIds[j].length - 1] = idx;
						final List<String> listToken = new ArrayList<>(listBatchesToken.get(0));
						listToken.set(listToken.size() - 1, tokenizer.decode(idx));
						listBatchesToken.add(listToken);
					}
				}
				else {
					inputIds = nextInputIds;
				}
			}
			System.out.println("Prompt: " + inputSentence);
			for(int batch = 0; batch < listBatchesToken.size(); batch++) {
				final String sResult = listBatchesToken.get(batch).stream().collect(Collectors.joining());
				System.out.println("Result " + batch + ":" + sResult);
			}
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
