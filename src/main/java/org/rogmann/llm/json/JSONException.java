package org.rogmann.llm.json;

/**
 * Exception which occurred while JSON-parsing.
 */
public class JSONException extends RuntimeException {
	/** serial number */
	private static final long serialVersionUID = 20230805L;

	/**
	 * Constructor
	 * @param message message
	 */
	public JSONException(String message) {
		super(message);
	}

	/**
	 * Constructor
	 * @param message message
	 * @param eCause cause
	 */
	public JSONException(String message, Exception eCause) {
		super(message, eCause);
	}
}
