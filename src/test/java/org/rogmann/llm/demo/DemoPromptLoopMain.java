package org.rogmann.llm.demo;

import java.io.File;
import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

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
 * This class receives prompt via standard-input.
 */
public class DemoPromptLoopMain {
	/** logger */
	private static final Logger LOG = Logger.getLogger(DemoPromptLoopMain.class.getName());

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

		ThreadProfiler profiler = null;
		if (useProfiler) {
			profiler = new ThreadProfiler(new Thread[] { Thread.currentThread() }, Long.MAX_VALUE, 0, 20, 5, System.out);
		}

		try (Scanner scanner = new Scanner(System.in)) {
			try (LlmExecutor executor = new LlmWorkerPoolPhaser(nThreads)) {
		
				// Set maxBatchSize = 3 to get three different beams.
				final int maxBatchSize = 1;
				final BloomModel model = new BloomModel(modelReader, maxBatchSize, executor);

				while (true) {
					System.out.println("Prompt: ");
					String inputSentence = scanner.nextLine();
					if (inputSentence == null || inputSentence.length() == 0) {
						break;
					}
					Instant tsStartInfer = Instant.now();
					final int maxToken = 60;
					final boolean useCache = Boolean.TRUE.booleanValue(); // use fusedQkv-cache?
					int[][] inputIds = tokenizer.encode(inputSentence);
					final int numTokenInput = inputIds[0].length;
					final List<List<String>> listBatchesToken = new ArrayList<>();
					listBatchesToken.add(new ArrayList<>());
					final float[][][][] layersFusedQkv = new float[model.getNumLayers()][maxBatchSize]
							[numTokenInput + maxToken][3 * model.getHiddenSize()];
					System.out.print("Response: ");
					for(int idxInf = 1; idxInf <= maxToken; idxInf++) {
						final int batchSize = inputIds.length;
		
						final float[][][] hiddenState;
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
							final float[][] batchState = hiddenState[b];
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
									idxCandidates.add(Integer.valueOf(idx));
								}
							}
							String sToken = tokenizer.decode(idx);
							System.out.print(sToken);
							final List<String> listToken = listBatchesToken.get(b);
							listToken.add(sToken);
							nextInputIds[b] = tokenizer.appendToken(inputIds[b], idx);
						}
		
						final int numCandidates = idxCandidates.size();
						inputIds = nextInputIds;
						final Instant tsEnd = Instant.now();
					}
					System.out.println();
					System.out.println();
				}
			}
		}
		
	}
}
