package org.rogmann.llm.json;

import java.util.regex.Pattern;

/**
 * Wrapper around a JSON-number
 */
public class JSONNumber {
	/** Pattern of JSON-number */
	static final Pattern P_NUMBER = Pattern.compile("-?(?:0|[1-9][0-9]*)(?:[.][0-9]+)?(?:[eE][+-][0-9]+)?");

	/** number */
	 private final String number;

	 /**
	  * Constructor
	  * @param number representation of number
	  */
	 public JSONNumber(String number) {
		 this.number = number;
	 }

	 /**
	  * Gets the JSON string representation of this number.
	  * @return string representation
	  */
	public String getNumberAsString() {
		return number;
	}
}
