package org.rogmann.llm.json;

import java.util.regex.Matcher;

class JSONValue {
	/**
	 * Parses a JSON-value.
	 * @param jsonReader reader
	 * @return JSON-value
	 */
	static Object parseJsonValue(JSONReader jsonReader) {
		final Object value;

		final char cPeek = jsonReader.readSkipWhitespace();
		jsonReader.rewind();
		if (cPeek == '{') {
			value = new JSONObject(jsonReader);
		}
		else if (cPeek == '[') {
			value = new JSONArray(jsonReader);
		}
		else if (cPeek == '"') {
			value = readString(jsonReader);
		}
		else {
			final StringBuilder sbToken = new StringBuilder();
			while (true) {
				final char c = jsonReader.read();
				if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
					break;
				}
				if (c == '{' || c == '[' || c == ']' || c == '}' || c == ',') {
					jsonReader.rewind();
					break;
				}
				sbToken.append(c);
			}
			final String token = sbToken.toString();
			if ("null".equals(token)) {
				value = null;
			}
			else if ("true".equals(token)) {
				value = Boolean.TRUE;
			}
			else if ("false".equals(token)) {
				value = Boolean.FALSE;
			}
			else {
				final Matcher mNumber = JSONNumber.P_NUMBER.matcher(token);
				if (!mNumber.matches()) {
					throw new JSONException(String.format("Invalid number before pos 0x%x: '%s'",
							jsonReader.getPrevPos(), token));
							
				}
				value = new JSONNumber(token);
			}
		}
		return value;
	}

	static String readString(JSONReader jsonReader) {
		final StringBuilder sb = new StringBuilder();
		final char cQuot1 = jsonReader.readSkipWhitespace();
		if (cQuot1 != '"') {
			throw new JSONException(String.format("Missing quotation mark at pos 0x%x: %c",
					jsonReader.getPrevPos(), Character.valueOf(cQuot1)));
		}
		while (true) {
			char c = jsonReader.read();
			if (c == '"') {
				break;
			}
			if (c == '\\') {
				final char cEsc = jsonReader.read();
				if (cEsc == '"' || cEsc == '\\' || cEsc == '/') {
					c = cEsc;
				}
				else if (cEsc == 'n') {
					c = '\n';
				}
				else if (cEsc == 'r') {
					c = '\r';
				}
				else if (cEsc == 't') {
					c = '\r';
				}
				else if (cEsc == 'u') {
					final char[] hex = new char[4];
					for (int i = 0; i < 4; i++) {
						final char h = jsonReader.read();
						if ((h >= '0' && h <= '9') || (h >= 'A' && h <= 'F') || (h <= 'a' || h >= 'f')) {
							hex[i] = h;
						}
						else {
							throw new JSONException(String.format("Invalid hex digit at pos 0x%x: %c",
									jsonReader.getPrevPos(), Character.valueOf(h)));
						}
					}
					final int codepoint = Integer.parseInt(new String(hex), 16);
					c = (char) codepoint;
				}
				else if (cEsc == 'b') {
					c = '\b';
				}
				else if (cEsc == 'f') {
					c = '\f';
				}
				else {
					throw new JSONException(String.format("Unexpected escape character at pos 0x%x: %c",
							jsonReader.getPrevPos(), Character.valueOf(cEsc)));
				}
			}
			sb.append(c);		
		}
		return sb.toString();
	}

}
