package org.rogmann.llm.json;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * JSON-object (map from string to JSON-value).
 */
public class JSONObject {
	/** logger */
	private static final Logger LOG = Logger.getLogger(JSONObject.class.getName());

	/** map from key to value */
	private final Map<String, Object> dict = new LinkedHashMap<>();

	/**
	 * Constructor
	 * @param jsonString string representation of JSON-object
	 */
	public JSONObject(String jsonString) {
		this(new JSONReader(jsonString));
	}

	/**
	 * Constructor
	 * @param jsonReader JSON-reader
	 */
	JSONObject(JSONReader jsonReader) {
		final char c = jsonReader.read();
		if (c != '{') {
			throw new JSONException(String.format("Unexpected start of JSONObject at pos 0x%x: %c",
					jsonReader.getPrevPos(), Character.valueOf(c)));
		}
		while (true) {
			final char cNext = jsonReader.readSkipWhitespace();
			if (cNext == '}') {
				break;
			}
			if (dict.size() == 0) {
				jsonReader.rewind();
			}
			else if (cNext != ',') {
				throw new JSONException(String.format("Unexpected character after value at pos 0x%x: %c",
						jsonReader.getPrevPos(), Character.valueOf(cNext)));
			}
			final String key = JSONValue.readString(jsonReader);
			final char cColon = jsonReader.readSkipWhitespace();
			if (cColon != ':') {
				throw new JSONException(String.format("Missing colon at pos 0x%x (after key '%s'): %c",
						jsonReader.getPrevPos(), key, Character.valueOf(cColon)));
			}
			final Object value = JSONValue.parseJsonValue(jsonReader);
			if (LOG.isLoggable(Level.FINEST)) {
				final String sValue = displayValue(value);
				LOG.finest(String.format("Key \"%s\" -> %s", key, sValue));
			}
			dict.put(key, value);
		}
	}

	/**
	 * Reads an integer.
	 * @param key key
	 * @return integer
	 * @throws JSONException in case of a missing entry or invalid number.
	 */
	public int getInt(String key) {
		final String sValue = getNumber(key);
		int value;
		try {
			value = Integer.parseInt(sValue);
		} catch (NumberFormatException e) {
			throw new JSONException(String.format("Invalid number format (%s) of key \"%s\", expected int.", sValue, key), e);
		}
		return value;
	}

	/**
	 * Reads a float.
	 * @param key key
	 * @return float
	 * @throws JSONException in case of a missing entry or invalid number.
	 */
	public float getFloat(String key) {
		final String sValue = getNumber(key);
		float value;
		try {
			value = Float.parseFloat(sValue);
		} catch (NumberFormatException e) {
			throw new JSONException(String.format("Invalid number format (%s) of key \"%s\", expected float.", sValue, key), e);
		}
		return value;
	}

	/**
	 * Reads a float.
	 * @param key key
	 * @return float
	 * @throws JSONException in case of a missing entry or <code>null</code>-value.
	 */
	public String getString(String key) {
		final Object oValue = getValue(key);
		if (!(oValue instanceof String)) {
			throw new JSONException(String.format("Invalid value %s of key \"%s\", string expected.", oValue.getClass().getName(), key));
		}
		return (String) oValue;
	}

	/**
	 * Reads a JSON-object.
	 * @param key key
	 * @return JSON-object
	 */
	public JSONObject getJSONObject(String key) {
		final Object oValue = getValue(key);
		if (!(oValue instanceof JSONObject)) {
			throw new JSONException(String.format("Invalid value %s of key \"%s\", JSONObject expected.", oValue.getClass().getName(), key));
		}
		return (JSONObject) oValue;
	}

	/**
	 * Gets <code>true</code> if the object contains the given key.
	 * @param key key
	 * @return <code>true</code> if key is known
	 */
	public boolean hasKey(String key) {
		return dict.containsKey(key);
	}

	/**
	 * Gets the internal value of a given key.
	 * @param key key
	 * @return internal value
	 * @throws JSONException if the key is missing or the value is <code>null</code>
	 */
	private Object getValue(String key) {
		final Object oValue = dict.get(key);
		if (oValue == null) {
			throw new JSONException(String.format("There is no value of key \"%s\".", key));
		}
		return oValue;
	}

	/**
	 * Gets the string representation of a number of a given key.
	 * @param key key
	 * @return string representation
	 */
	private String getNumber(String key) {
		final Object oValue = getValue(key);
		if (!(oValue instanceof JSONNumber)) {
			throw new JSONException(String.format("Invalid value %s of key \"%s\", number expected.", oValue.getClass().getName(), key));
		}
		final String sValue = ((JSONNumber) oValue).getNumberAsString();
		return sValue;
	}

	/**
	 * Gets the size of the JSON-object
	 * @return number of entries
	 */
	public int length() {
		return dict.size();
	}

	/**
	 * Gets the key-set of the JSON-object.
	 * @return key-set
	 */
	public Set<String> keySet() {
		return dict.keySet();
	}

	/**
	 * Gives information about a value.
	 * @param value value
	 * @return short text
	 */
	private String displayValue(final Object value) {
		final String sValue;
		if (value == null) {
			sValue = "null";
		}
		else if (value instanceof String) {
			sValue = "string of length " + ((String) value).length();
		}
		else if (value instanceof JSONNumber) {
			sValue = "number " + ((JSONNumber) value).getNumberAsString();
		}
		else if (value instanceof JSONObject) {
			sValue = "JSON-object of size " + ((JSONObject) value).dict.size();
		}
		else if (value instanceof JSONArray) {
			sValue = "JSON-array of length " + ((JSONArray) value).length();
		}
		else if (value instanceof Boolean) {
			sValue = "boolean " + ((Boolean) value).toString();
		}
		else {
			throw new JSONException("Unexpected value of type " + value.getClass());
		}
		return sValue;
	}

}
