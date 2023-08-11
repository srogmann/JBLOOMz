package org.rogmann.llm.tokenizer;

/**
 * Interface of a tokenizer.
 */
public interface Tokenizer {
	/**
	 * Gets the number of tokens.
	 * @return number
	 */
	int size();

	/**
	 * Encodes a strings.
	 * @param s string
	 * @return array of length one containing the tokenized string
	 */
	int[][] encode(String s);
	
	/**
	 * Decodes a token.
	 * @param idx index of the token
	 * @return string
	 */
	String decode(int idx);

	/**
	 * Appends a token to a given array of input-ids.
	 * @param inputIds input-ids
	 * @param idx token to be added
	 * @return new input-ids
	 */
	int[][] appendToken(int[][] inputIds, int idx);

	/**
	 * Convert token into internal representation (e.g. byte-level).
	 * @param token token
	 * @return internal representation
	 */
	String convertToInternal(String token);

	/**
	 * Converts token from interal representation (e.g. byte-level).
	 * @param s internal representation
	 * @return token
	 */
	String convertFromInternal(String s);

	/**
	 * Gets the version of the tokenizer.
	 * @return version
	 */
	String getVersion();

}
