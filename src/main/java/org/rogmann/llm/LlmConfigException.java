package org.rogmann.llm;

/**
 * Exception in the configuration of the model.
 */
public class LlmConfigException extends Exception {
	/** serial number */
	private static final long serialVersionUID = 20230728L;
	
	/**
	 * Constructor
	 * @param message message
	 */
	public LlmConfigException(String message) {
		super(message);
	}

	/**
	 * Constructor
	 * @param message message
	 * @param eCause cause
	 */
	public LlmConfigException(String message, Throwable eCause) {
		super(message, eCause);
	}

}
