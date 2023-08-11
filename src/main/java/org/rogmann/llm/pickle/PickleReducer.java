package org.rogmann.llm.pickle;

import org.rogmann.llm.LlmConfigException;

/**
 * Interface of a pickle-reducer, an implementation of REDUCE.
 */
public interface PickleReducer {

	/**
	 * Reduces an object using arguments.
	 * @param obj object to be reduced
	 * @param args arguments
	 * @return reduced object (<code>null</code> can be represented by Void.TYPE) or <code>null</code>
	 * @throws LlmConfigException in case of a configuration-error (or OOME)
	 */
	Object reduce(final Object obj, Object[] args) throws LlmConfigException;

	/**
	 * Finish building an object.
	 * @param obj object
	 * @param args arguments, e.g. a HashMap
	 * @return finished object
	 */
	Object build(Object obj, Object args);

}
