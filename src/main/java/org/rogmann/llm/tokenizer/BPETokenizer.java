package org.rogmann.llm.tokenizer;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.rogmann.llm.json.JSONObject;

/**
 * Class to convert a text into tokens (Byte-Pair-Encoding).
 *  
 * <p>This class implements a byte-level BPE (Byte-Pair Encoding).</p>
 */
public class BPETokenizer implements Tokenizer {
	/** byte level mapping from byte to char */
	private static final char[] BYTE_TO_CHAR;
	/** byte level mapping from char to byte */
	private static final byte[] CHAR_TO_BYTE;

	/** JSON-tokenizer */
	private final JSONObject fJson;

	/** version of the tokenizer */
	private final String fVersion;

	/** model */
	private final JSONObject fJsonModel;

	/** map from token to index */
	private final Map<String, Integer> mapTokenIdx = new HashMap<>();

	/** map from index to token */
	private final Map<Integer, String> mapIdxToken = new HashMap<>();

	/** List of tokens */
	private final String[] fToken;

	/**
	 * Static initializer of byte-level mapping:
	 * See function bytes_char() in https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs.
	 * 
	 * <p>Bytes are mapped to printable characters.</p>
	 */
	static {
		BYTE_TO_CHAR = new char[256];
		CHAR_TO_BYTE = new byte[0x144];
		boolean[] bPrintable = new boolean[256];
		Arrays.fill(bPrintable, '!', '~' + 1, true);
		Arrays.fill(bPrintable, 0xa1, 0xac + 1, true);
		Arrays.fill(bPrintable, 0xae, 0xff + 1, true);
		int n = 0;
		for (int b = 0; b < 256; b++) {
			if (bPrintable[b]) {
				BYTE_TO_CHAR[b] = (char) b;
				CHAR_TO_BYTE[b] = (byte) b;
			}
			else {
				BYTE_TO_CHAR[b] = (char) (0x100 + n);
				CHAR_TO_BYTE[0x100 + n] = (byte) b;
				n++;
			}
		}
	}

	/**
	 * Constructor
	 * @param is input-stream of JSON-tokenizer
	 * @throws IOException in case of an IO-error
	 */
	public BPETokenizer(final InputStream is) throws IOException {
		final StringBuilder sb = new StringBuilder(500);
		final char[] cBuf = new char[1024];
		try (InputStreamReader isr = new InputStreamReader(is, StandardCharsets.UTF_8)) {
			while (true) {
				final int len = isr.read(cBuf);
				if (len == -1) {
					break;
				}
				sb.append(cBuf, 0, len);
			}
		}
		fJson = new JSONObject(sb.toString());
		fVersion = fJson.getString("version");
		fJsonModel = fJson.getJSONObject("model");
		final JSONObject jsonVocab = fJsonModel.getJSONObject("vocab");
		fToken = new String[jsonVocab.length()];
		for (String token : jsonVocab.keySet()) {
			final int idx = jsonVocab.getInt(token);
			fToken[idx] = token;
			mapTokenIdx.put(token, Integer.valueOf(idx));
			mapIdxToken.put(Integer.valueOf(idx), token);
		}
	}

	/** {@inheritDoc] */
	@Override
	public int size() {
		return mapTokenIdx.size();
	}

	/** {@inheritDoc] */
	@Override
	public int[][] encode(String s) {
		final List<Integer> list = new ArrayList<>();
		final int len = s.length();
		int i = 0;
		while (i < len) {
			final String sPart = convertToInternal(s.substring(i)); // ' ' is mapped to 'Ġ'.
			String tmpToken = null;
			for (int idx = 0; idx < fToken.length; idx++) {
				final String t = fToken[idx];
				if (sPart.startsWith(t) && (tmpToken == null || t.length() > tmpToken.length())) {
					tmpToken = t;
				}
			}
			for (int idx = 0; idx < fToken.length; idx++) {
				final String t = fToken[idx];
				if (sPart.startsWith(t) && (tmpToken == null || t.length() > tmpToken.length())) {
					tmpToken = t;
				}
			}
			if (tmpToken == null) {
				throw new RuntimeException(String.format("Can't tokenize: \"%s\"", sPart));
			}
			final Integer iIdx = mapTokenIdx.get(tmpToken);
			list.add(iIdx);
			i += convertFromInternal(tmpToken).length();
		}
		final int[] aIds = new int[list.size()];
		for (int j = 0; j < aIds.length; j++) {
			aIds[j] = list.get(j);
		}
		final int[][] output = new int[1][aIds.length];
		output[0] = aIds;
		return output;
	}

	/** {@inheritDoc] */
	@Override
	public String decode(int idx) {
		String blToken = mapIdxToken.get(Integer.valueOf(idx));
		return convertFromInternal(blToken); // 'Ġ' maps to ' '.
	}

	/** {@inheritDoc] */
	@Override
	public String convertToInternal(String s) {
		final byte[] buf = s.getBytes(StandardCharsets.UTF_8);
		final char[] cBuf = new char[buf.length];
		for (int i = 0; i < buf.length; i++) {
			cBuf[i] = BYTE_TO_CHAR[buf[i] & 0xff];
		}
		return new String(cBuf);
	}

	/** {@inheritDoc] */
	@Override
	public String convertFromInternal(String s) {
		final byte[] buf = new byte[s.length()];
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c >= CHAR_TO_BYTE.length) {
				throw new IllegalArgumentException(String.format("Invalid character 0x%04x", c));
			}
			buf[i] = CHAR_TO_BYTE[c];
			if (buf[i] == 0x00) {
				throw new IllegalArgumentException(String.format("Invalid character 0x%04x (maps to 0x00)", 
						Integer.valueOf(c)));
			}
		}
		return new String(buf, StandardCharsets.UTF_8);
	}

	/** {@inheritDoc] */
	@Override
	public String getVersion() {
		return fVersion;
	}

}
