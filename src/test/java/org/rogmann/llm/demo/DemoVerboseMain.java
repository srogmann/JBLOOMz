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
import org.rogmann.llm.visualize.ExportPPM;

/**
 * Executes a text generation using a BLOOM based model.
 */
public class DemoVerboseMain {
	/** logger */
	private static final Logger LOG = Logger.getLogger(DemoVerboseMain.class.getName());
	
	/** optional file-name to export fusedQkv-layers as portable bitmap */
	private static final String FILENAME_FUSED_QKV_PPM = System.getProperty("jbloomz.fusedqkv.ppm.file");

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
	
			// Set maxBatchSize = 3 to get three different beams.
			final int maxBatchSize = 1;
			final BloomModel model = new BloomModel(modelReader, maxBatchSize, executor);
	
			tsStartInfer = Instant.now();
			//String inputSentence = "Auf der Wiese läuft ein Hund hinter";
			//String inputSentence = "Der Hund heißt Karl. Die Katze heißt Mimi. Wie nennt Mimi den Hund?";
			String inputSentence = "Translate to Chinese: I write a program in Java.";
			//String inputSentence = "What is the capital of France?";
			//String inputSentence = "Translate to chinese: cat.";
			//String inputSentence = "¿Quién era Joan Miró?";
			final int maxToken = 7;
			final boolean useCache = Boolean.TRUE.booleanValue(); // use fusedQkv-cache?
			int[][] inputIds = tokenizer.encode(inputSentence);
			final int numTokenInput = inputIds[0].length;
			final int numBeams = maxBatchSize;
			if (profiler != null) {
				profiler.start();
			}
			final List<List<String>> listBatchesToken = new ArrayList<>();
			listBatchesToken.add(new ArrayList<>());
			final float[][][][] layersFusedQkv = new float[model.getNumLayers()][maxBatchSize]
					[numTokenInput + maxToken][3 * model.getHiddenSize()];
			for(int idxInf = 1; idxInf <= maxToken; idxInf++) {
				System.out.println("");
				System.out.println("Inference " + idxInf);
				final int batchSize = inputIds.length;
				for (int b = 0; b < batchSize; b++) {
					System.out.println(String.format("input_ids[%d]: %s", b, Arrays.toString(inputIds[b])));
				}
				System.out.println("Start: " + LocalDateTime.now());

				final float[][][][] hiddenState;
				if (idxInf == 1 || !useCache) {
					// First iteration, computes the fusedQkv-entries of the input-tokens.
					hiddenState = model.forward(inputIds, layersFusedQkv, null);
				}
				else {
					// Next iteration, re-use of existing fusedQkv-entries.
					final int curNumSeqIdx = numTokenInput + idxInf - 1;
					final Integer numSeqLenCache = Integer.valueOf(curNumSeqIdx - 1);
					final int[][] inputIdsForward = new int[batchSize][1];
					for (int b = 0; b < batchSize; b++) {
						inputIdsForward[b][0] = inputIds[b][curNumSeqIdx - 1];
					}
					hiddenState = model.forward(inputIdsForward, layersFusedQkv, numSeqLenCache);
				}

				final List<Integer> idxCandidates = new ArrayList<>();
				final int[][] nextInputIds = new int[batchSize][];
				for (int b = 0; b < batchSize; b++) {
					System.out.println("Batch " + b);
					final float[][] batchState = hiddenState[model.getNumLayers()][b];
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
					final List<String> listToken = listBatchesToken.get(b);
					listToken.add(sToken);
					nextInputIds[b] = tokenizer.appendToken(inputIds[b], idx);
				}

				final int numCandidates = idxCandidates.size();
				if (idxInf == 2 && numCandidates >= numBeams && numBeams > batchSize) {
					inputIds = new int[numBeams][];
					inputIds[0] = nextInputIds[0];
					System.out.println("Switch to batch: #beams=" + numBeams);
					for (int b = 1; b < numBeams; b++) {
						inputIds[b] = nextInputIds[0].clone();
						final int idx = idxCandidates.get(numCandidates - 1 - b).intValue();
						final String currToken = tokenizer.decode(idx);
						System.out.println(String.format("Beam %d: token %d (%s)",
								b, idx, currToken));
						inputIds[b][inputIds[b].length - 1] = idx;
						final List<String> listToken = new ArrayList<>(listBatchesToken.get(0));
						listToken.set(listToken.size() - 1, currToken);
						listBatchesToken.add(listToken);
						for (int l = 0; l < model.getNumLayers(); l++) {
							for (int j = 0; j < numTokenInput + maxToken; j++) {
								System.arraycopy(layersFusedQkv[l][0][j], 0, layersFusedQkv[l][b][j], 0, 3 * model.getHiddenSize());
							}
						}
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
			if (FILENAME_FUSED_QKV_PPM != null) {
				ExportPPM.exportFusedQkv(new File(FILENAME_FUSED_QKV_PPM), layersFusedQkv);
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
