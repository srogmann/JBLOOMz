package org.rogmann.llm.bloom;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llm.LlmConfigException;
import org.rogmann.llm.LlmExecutor;
import org.rogmann.llm.ModelReader;
import org.rogmann.llm.ModelReaderBinary;
import org.rogmann.llm.nn.Embeddings;
import org.rogmann.llm.nn.LayerNorm;
import org.rogmann.llm.nn.PickleReducerTorch;
import org.rogmann.llm.nn.Tensor;
import org.rogmann.llm.nn.TensorProvider;
import org.rogmann.llm.pickle.PickleReader;
import org.rogmann.llm.tokenizer.Tokenizer;

public class BloomModel implements TensorProvider {
	/** Logger */
	private static final Logger LOG = Logger.getLogger(BloomModel.class.getName());

	/** number of layers */
	private final int numLayers;
	/** number of attention-heads */
	private final int numHeads;
	/** hidden size */
	private final int hiddenSize;

	/** maximum batch-size */
	protected final int maxBatchSize;
	
	/** map from key to tensor */
	private final Map<String, Tensor> mapTensors = new HashMap<>(100);
	
	/** key-prefix, e.g. "" or "transformer." */
	private final String keyPrefix;

	/** embeddings of token in hidden states */
	private final Embeddings embeddings;

	private final LayerNorm wordEmbeddingsLayerNorm;
	
	private final BloomBlock[] blocks;

	private final LayerNorm lnF;

	private final LlmExecutor executor;

	public BloomModel(ModelReader modelReader, int maxBatchSize, LlmExecutor executor) throws IOException, LlmConfigException {
		this.numLayers = modelReader.nLayer;
		this.numHeads = modelReader.nHead;
		this.hiddenSize = modelReader.hiddenSize;
		LOG.info(String.format("Read %s-model '%s' with %d %s, %d %s and hidden size %d",
				modelReader.modelType, modelReader.getModelName(),
				Integer.valueOf(numLayers), (numLayers == 1) ? "layer" : "layers",
				Integer.valueOf(numHeads), (numHeads == 1) ? "head" : "heads",
				Integer.valueOf(hiddenSize)));
		this.executor = executor;
		this.maxBatchSize = maxBatchSize;
		
		String keyPrefix = "";
		
		List<File> modelFiles = modelReader.getPytorchModelFile();
		for (File modelFile : modelFiles) {
			LOG.info("Read model-file " + modelFile.getName());
			try (ModelReaderBinary readerBinary = new ModelReaderBinary(modelFile, modelReader.supportUnpacked())) {
				final Map<String, Object> result;
				try (BufferedInputStream bis = new BufferedInputStream(readerBinary.getAsStream("data.pkl"))) {
					PickleReader reader = new PickleReader(bis, new PickleReducerTorch(executor));
					@SuppressWarnings("unchecked")
					final Map<String, Object> mResult = reader.getResult(Map.class);
					result = mResult;
				}
				LOG.info("pickle-map.size: " + result.size());
				for (Entry<String, Object> entry : result.entrySet()) {
					final String key = entry.getKey();
					final Object oValue = entry.getValue();
					if (oValue instanceof Tensor) {
						final Tensor tensor = (Tensor) oValue;
						try {
							tensor.readTensorData(key, readerBinary);
						} catch (LlmConfigException e) {
							throw new LlmConfigException("Configuration error when reading tensor " + key, e);
						}
						mapTensors.put(key, tensor);
						if (key.startsWith("transformer.")) {
							keyPrefix = "transformer.";
						}
					}
				}
			}
			catch (IOException e) {
				throw new IOException("IO-error while reading " + modelFiles, e);
			}
			finally {
				LOG.info(String.format("Total memory: %.1f MB", Runtime.getRuntime().totalMemory() / 1048576.0));
				LOG.info(String.format("Free memory: %.1f MB", Runtime.getRuntime().freeMemory() / 1048576.0));
			}
		}
		this.keyPrefix = keyPrefix;
		
		final float[][] tWeights = get("word_embeddings.weight").t2;
		embeddings = new Embeddings(tWeights, executor);

		final float[] tWordEmbeddingsLayernomWeight = get("word_embeddings_layernorm.weight").t1;
		final float[] tWordEmbeddingsLayernomBias = get("word_embeddings_layernorm.bias").t1;
		LOG.finer("we.weight: " + Arrays.toString(Arrays.copyOfRange(tWordEmbeddingsLayernomWeight, 0, 5)));
		LOG.finer("we.bias:   " + Arrays.toString(Arrays.copyOfRange(tWordEmbeddingsLayernomBias, 0, 5)));
		wordEmbeddingsLayerNorm = new LayerNorm(1e-5f, tWordEmbeddingsLayernomWeight, tWordEmbeddingsLayernomBias);

		blocks = new BloomBlock[numLayers];
		for (int i = 0; i < numLayers; i++) {
			final int layer = i;
			LOG.info("Load Layer " + layer);
			try {
				blocks[i] = new BloomBlock(maxBatchSize, hiddenSize, numHeads, i,
					this, executor);
			} catch (IOException e) {
				throw new IOException("IO-exception while reading block of layer " + i, e);
			}
		}

		final float[] tLnFWeight = get("ln_f.weight").t1;
		final float[] tLnFBias = get("ln_f.bias").t1;
		lnF = new LayerNorm(1e-5f, tLnFWeight, tLnFBias);

	}

	/**
	 * Gets the embeddings of the token into the hidden states.
	 * @return embeddings
	 */
	public Embeddings getEmbeddings() {
		return embeddings;
	}

	/**
	 * Gets the number of layers of the transformer model.
	 * @return number of layers
	 */
	public int getNumLayers() {
		return numLayers;
	}

	/**
	 * Gets the dimension of the hidden size.
	 * @return hidden size
	 */
	public int getHiddenSize() {
		return hiddenSize;
	}

	/**
	 * Reads a configured tensor.
	 * @param key key of the tensor
	 * @return tensor
	 * @throws LlmConfigException in case of an unknown tensor
	 */
	public Tensor get(String key) throws LlmConfigException {
		String keyMap = keyPrefix + key;
		Tensor tensor = mapTensors.get(keyMap);
		if (tensor == null) {
			LOG.info("Known keys: " + new TreeSet<>(mapTensors.keySet()));
			throw new LlmConfigException("No tensor with keyMap " + key + " for key " + key);
		}
		if (LOG.isLoggable(Level.FINE)) {
			LOG.fine(String.format("Tensor %s: %s", key, Arrays.toString(tensor.getShape())));
		}
		return tensor;
	}

	/**
	 * Executes the model.
	 * @param inputIds input-ids (batchSize, numSeq)
	 * @return hidden states (batchSize, seqLen, dim of weights)
	 */
	public float[][][] forward(final int[][] inputIds) {
		final int batchSize = inputIds.length;
		final int numSeq = inputIds[0].length;
		final float[][][] fusedQkv = new float[batchSize][numSeq][3 * hiddenSize];
		float[][][][] layersFusedQkv = new float[numLayers][][][];
		// We use the same temporary tensor in each layer.
		Arrays.fill(layersFusedQkv, fusedQkv);
		final Integer numSeqLenCache = null;
		return forward(inputIds, layersFusedQkv, numSeqLenCache)[numLayers];
	}

	/**
	 * Executes the model.
	 * @param inputIds input-ids (batchSize, numSeq)
	 * @param layersFusedQkv fusedQkv-tensor to be used in attention-computation (numLayers, batchSize, numSeq, 3 * hiddenSize)
	 * @param numSeqLenCache <code>null</code> if no cache is used, numSeq in fusedQkv otherwise
	 * @return hidden states (layers + 1, batchSize, seqLen, dim of weights)
	 */
	public float[][][][] forward(final int[][] inputIds,
			final float[][][][] layersFusedQkv, final Integer numSeqLenCache) {
		final int batchSize = inputIds.length;
		final int seqLen = inputIds[0].length;
		final int totalSeqLen = (numSeqLenCache == null) ? seqLen : numSeqLenCache + seqLen;

		float[][][] inputEmbeds = embeddings.wordEmbeddings(inputIds);
		if (LOG.isLoggable(Level.FINE)) {
			for (float[] row : inputEmbeds[0]) {
				LOG.fine("InputEmbeds, Row: " + Arrays.toString(Arrays.copyOfRange(row, 0, 3)));
			}
		}

		// hidden states is a tensor of shape (batchSize, inputSize, hiddenSize).
		final float[][][][] hiddenStates = new float[numLayers + 1][batchSize][seqLen][hiddenSize];
		for (int idxI = 0; idxI < batchSize; idxI++) {
			final int i = idxI;
			executor.startLoopTasks(seqLen, (jStart, jEnd) -> () -> {
				for (int j = 0; j < seqLen; j++) {
					hiddenStates[0][i][j] = wordEmbeddingsLayerNorm.normalize(inputEmbeds[i][j]);
				}
			});
		}
		if (LOG.isLoggable(Level.FINER)) {
			for (float[] row : hiddenStates[0][0]) {
				LOG.finer("Hidden-Row: " + Arrays.toString(Arrays.copyOfRange(row, 0, 3)));
			}
		}
		
		final float[][] attentionMask = new float[batchSize][totalSeqLen];
		for (int i = 0; i < batchSize; i++) {
			Arrays.fill(attentionMask[i], 1.0f);
		}

		final Tensor alibi = BloomAlibi.buildAlibiTensor(attentionMask, numHeads, executor);
		if (LOG.isLoggable(Level.FINER)) {
			for (float[][] aTmp : alibi.t3) {
				LOG.finer("ALiBi row: " + Arrays.toString(aTmp[0]));
			}
		}

		boolean[][][][] causalMask = new boolean[batchSize][1][totalSeqLen][totalSeqLen];
		for (boolean[][][] batchCM : causalMask) {
			for (int i = 0; i < totalSeqLen; i++) {
				for (int j = i + 1; j < totalSeqLen; j++) {
					batchCM[0][i][j] = true;
				}
			}
		}

		final float[][][] attentionResidual = new float[batchSize][seqLen][hiddenSize];
		for(int layer = 0; layer < numLayers; layer++) {
			LOG.fine("Compute Layer " + layer);
			blocks[layer].forward(hiddenStates[layer],
					layersFusedQkv[layer], numSeqLenCache,
					causalMask, alibi, attentionResidual, hiddenStates[layer + 1]);
		}

		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < seqLen; j++) {
				hiddenStates[numLayers][i][j] = lnF.normalize(hiddenStates[numLayers][i][j]);
			}
		}
		if (LOG.isLoggable(Level.FINER)) {
			for (float[] row : hiddenStates[0][0]) {
				LOG.finer("Last normalize: " + Arrays.toString(Arrays.copyOfRange(row, 0, 3)));
			}
		}

		return hiddenStates;
	}

	/**
	 * Text generation: Computes the next tokens.
	 * @param tokenizer LLM-tokenizer
	 * @param model LLM-model
	 * @param inputSentence input
	 * @param maxToken maximum number of tokens to be generated
	 * @return generated list of tokens
	 */
	public List<String> computeNextTokens(Tokenizer tokenizer, BloomModel model, String inputSentence, int maxToken) {
		int[][] inputIds = tokenizer.encode(inputSentence);
		final List<String> listToken = new ArrayList<>();
		for(int idxInf = 1; idxInf <= maxToken; idxInf++) {
			final int[] aIdx = new int[inputIds.length];
			final float[][][] hiddenState = forward(inputIds);
			for (int batch = 0; batch < inputIds.length; batch++) {
				LOG.info("Batch " + batch + ", Inference " + idxInf);
				final float[][] batchState = hiddenState[batch];
				final int idx = model.getEmbeddings().computeMaxToken(batchState, tokenizer);
				final String sToken = tokenizer.decode(idx);
				aIdx[batch] = idx;
				LOG.info("Token: " + sToken);
				if (batch == 0) {
					// We collect the tokens of the first batch-element only.
					listToken.add(sToken);
				}
				inputIds[batch] = tokenizer.appendToken(inputIds[batch], idx);
			}		
		}
		return listToken;
	}

}
