package org.rogmann.llm;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.rogmann.llm.json.JSONException;
import org.rogmann.llm.json.JSONObject;

/**
 * This class reads a model saved by torch by huggingface.
 * It understands a subset of models only, e.g. BLOOM.
 */
public class ModelReader {

	/** model-folder */
	protected final File folder;
	/** <code>true</code> if the reader should look for unzipped model-bin-data */
	private final boolean supportUnpacked;
	/** JSON-configuration of the model */
	protected final JSONObject fConfigJson;

	public final int hiddenSize;
	public final float layerNormEpsilon;
	public final String modelType;
	/** number of layers (decoder-instances) in the model */
	public final int nLayer;
	/** number of attention-heads */
	public final int nHead;
	public final int offsetAlibi;
	//spublic final int seqLength;
	public final int unkTokenId;
	public final int vocabSize;
	public final String transformersVersion;

	/** file(s) containing the weights of the model (large split models are not supported yet) */	
	private final List<File> filesModelBin = new ArrayList<>();

	/**
	 * Constructor
	 * @param folder model-folder
	 * @param supportUnpacked <code>true</code> if the reader should look for unzipped model-bin-data
	 * @throws IOException in case of an IO-error
	 * @throws LlmConfigException configuration-error
	 */
	public ModelReader(File folder, boolean supportUnpacked) throws IOException, LlmConfigException {
		this.folder = folder;
		this.supportUnpacked = supportUnpacked;
		if (!folder.isDirectory()) {
			throw new IOException("model folder is not a directory: " + folder);
		}
		final File[] files = folder.listFiles();
		if (files == null) {
			throw new IOException("model folder is missing: " + folder);
		}


		final File fileConfig = new File(folder, "config.json");
		try {
			fConfigJson = readJsonFile(fileConfig);
			// Transformer 4.20.0 used n_embed instead of hidden_size.
			hiddenSize = fConfigJson.hasKey("hidden_size") ? readInt("hidden_size") : readInt("n_embed");
			layerNormEpsilon = readFloat("layer_norm_epsilon");
			modelType = readString("model_type");
			// Transformer 4.20.0 used num_attention_heads instead of n_head.
			nHead = fConfigJson.hasKey("n_head") ? readInt("n_head") : readInt("num_attention_heads");
			nLayer = readInt("n_layer");
			//offsetAlibi = readInt("offset_alibi");
			offsetAlibi = 100;
			//seqLength = readInt("seq_length");
			// unkTokenId = readInt("unk_token_id");
			unkTokenId = 0;
			transformersVersion = readString("transformers_version");
			vocabSize = readInt("vocab_size");
		} catch (LlmConfigException e) {
			throw new LlmConfigException("configuration-error while reading config-file " + fileConfig, e);
		}
		
		final File fileModelBin = new File(folder, "pytorch_model.bin");
		if (fileModelBin.isFile()) {
			filesModelBin.add(fileModelBin);
		}
		else {
			final Pattern pPytorchModel = Pattern.compile("pytorch_model-([0-9]{1,6})-of-([0-9]{1,6}).bin");
			for (File file : files) {
				if (file.isDirectory()) {
					continue;
				}
				final Matcher mFile = pPytorchModel.matcher(file.getName());
				if (mFile.matches()) {
					filesModelBin.add(file);
				}
			}
		}
		if (filesModelBin.size() == 0) {
			throw new IOException("model binary file is missing: " + fileModelBin);
		}
	}

	/**
	 * Gets the name of the model.
	 * @return model-name
	 */
	public String getModelName() {
		return folder.getName();
	}

	/**
	 * Gets the file(s) containing the weights of the model.
	 * @return file or list of files
	 */
	public List<File> getPytorchModelFile() {
		return filesModelBin;
	}

	private int readInt(String key) throws LlmConfigException {
		try {
			return fConfigJson.getInt(key);
		} catch (JSONException e) {
			throw new LlmConfigException("Error in config-key " + key, e);
		}
	}

	private float readFloat(String key) throws LlmConfigException {
		try {
			return fConfigJson.getFloat(key);
		} catch (JSONException e) {
			throw new LlmConfigException("Error in config-key " + key);
		}
	}

	private String readString(String key) throws LlmConfigException {
		try {
			return fConfigJson.getString(key);
		} catch (JSONException e) {
			throw new LlmConfigException("Error in config-key " + key);
		}
	}

	private JSONObject readJsonFile(File file) throws IOException {
		final StringBuilder sb = new StringBuilder(500);
		final char[] cBuf = new char[1024];
		try (InputStreamReader isr = new InputStreamReader(new BufferedInputStream(new FileInputStream(file)),
				StandardCharsets.UTF_8)) {
			while (true) {
				final int len = isr.read(cBuf);
				if (len == -1) {
					break;
				}
				sb.append(cBuf, 0, len);
			}
		}
		catch (IOException e) {
			throw new IOException("IO-error while reading json-file " + file, e);
		}
		return new JSONObject(sb.toString());
	}

	/**
	 * Gets <code>true</code> if the reader should look for unzipped model-bin-files.
	 * @return flag
	 */
	public boolean supportUnpacked() {
		return supportUnpacked;
	}

}
