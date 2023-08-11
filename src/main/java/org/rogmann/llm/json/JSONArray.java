package org.rogmann.llm.json;

import java.util.ArrayList;
import java.util.List;

/**
 * JSON-array.
 */
public class JSONArray {
	/** array-elements */
	private final List<Object> list;

	/**
	 * Constructor
	 * @param jsonReader JSON-reader
	 */
	JSONArray(JSONReader jsonReader) {
		list = new ArrayList<>();
	
		final char c = jsonReader.read();
		if (c != '[') {
			throw new JSONException(String.format("Unexpected start of JSONArray at pos %d: %c",
					jsonReader.getPrevPos(), Character.valueOf(c)));
		}
		while (true) {
			final char cNext = jsonReader.readSkipWhitespace();
			if (cNext == ']') {
				break;
			}
			if (list.size() == 0) {
				jsonReader.rewind();
			}
			else if (cNext != ',') {
				throw new JSONException(String.format("Unexpected character after value in JSONArray at pos %d: %c",
						jsonReader.getPrevPos(), Character.valueOf(cNext)));
			}
			final Object value = JSONValue.parseJsonValue(jsonReader);
			list.add(value);
		}
		
	}

	/**
	 * Gets the length of the JSON-array.
	 * @return number of elements
	 */
	public int length() {
		return list.size();
	}

}
