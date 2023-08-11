package org.rogmann.llm.json;

/**
 * Internal reader.
 */
class JSONReader {
	/** JSON-string to be parsed */
	private final String jsonString;
	/** length of the string */
	final int len;
	/** current position */
	int curPos;

	JSONReader(String jsonString) {
		this.jsonString = jsonString;
		len = jsonString.length();
		curPos = 0;
	}
	
	/**
	 * Peeks the next character in the stream but does not move the current position.
	 * @return next character
	 */
	public char peek() {
		if (curPos >= len) {
			throw new JSONException("Unexpected end of stream: " + curPos);
		}
		return jsonString.charAt(curPos);
	}

	/**
	 * Moves the position to the previous character.
	 */
	public void rewind() {
		curPos--;
	}

	/**
	 * Reads the next character in the stream.
	 * @return next character
	 */
	public char read() {
		if (curPos >= len) {
			throw new JSONException("Unexpected end of stream: " + curPos);
		}
		return jsonString.charAt(curPos++);
	}

	/**
	 * Reads the next character in the stream, skip whitespace.
	 * @return next character (non-whitespace)
	 */
	public char readSkipWhitespace() {
		if (curPos >= len) {
			throw new JSONException("Unexpected end of stream: " + curPos);
		}
		char c = jsonString.charAt(curPos++);
		while (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
			if (curPos >= len) {
				throw new JSONException("Unexpected end of stream: " + curPos);
			}
			c = jsonString.charAt(curPos++);
		}
		return c;
	}

	/**
	 * Gets the previous position.
	 * @return previous position
	 */
	public Integer getPrevPos() {
		return Integer.valueOf(curPos - 1);
	}
}
