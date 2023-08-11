package org.rogmann.llm.nn;

import org.rogmann.llm.LlmConfigException;

/**
 * Format of a floating-point number in memory.
 */
public enum StorageFormat {

	/** half precision, binary16, exponent uses 5 bits */
	FLOAT16("HalfStorage", 2),
	/** half precision, bfloat16 (brain floating point), exponent uses 8 bits */
	BFLOAT16("BFloat16Storage", 2),
	/** single precision, binary32, exponent uses 8 bits */
	FLOAT32("FloatStorage", 4);

	/** name of the format used in pickle-files of torch */
	public final String torchName;

	/** size of a number in bytes */
	public final int size;

	private StorageFormat(String torchName, int size) {
		this.torchName = torchName;
		this.size = size;
	}

	/**
	 * Lookup of a storage-format.
	 * @param name torch-name
	 * @return format
	 * @throws LlmConfigException in case of an unknown format
	 */
	public static StorageFormat lookupByTorchName(String name) throws LlmConfigException {
		for (StorageFormat format : values()) {
			if (format.torchName.equals(name)) {
				return format;
			}
		}
		throw new LlmConfigException("Unknown torch number-format " + name);
	}
}
