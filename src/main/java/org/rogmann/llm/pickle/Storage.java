package org.rogmann.llm.pickle;

/**
 * Definition of a tensor (format, location).
 */
public class Storage {

	public final PickleGlobal type;
	public final String key;
	public final String location;
	public final int size;
	
	Storage(PickleGlobal type, String key, String location, int size) {
		this.type = type;
		this.key = key;
		this.location = location;
		this.size = size;
	}

	public String toString() {
		return String.format("storage:{type:%s, key:%s, location:%s, size:%d}",
				type, key, location, Integer.valueOf(size));
	}
}
