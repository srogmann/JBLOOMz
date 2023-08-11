package org.rogmann.llm.nn;

import org.rogmann.llm.LlmConfigException;

/** Interface used to get a tensor */
public interface TensorProvider {

	/**
	 * Gets a configured and initialized tensor.
	 * @param key key of the tensor
	 * @return tensor
	 * @throws LlmConfigException in case of an unknown tensor
	 */
	Tensor get(String key) throws LlmConfigException;
}
