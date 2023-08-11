package org.rogmann.llm.pickle;

/**
 * global-object of pickle.
 */
public class PickleGlobal {
	/** module-name */
	public final String moduleName;
	/** class-name */
	public final String className;

	/**
	 * Constructor
	 * @param moduleName module-name
	 * @param className class-name
	 */
	PickleGlobal(String moduleName, String className) {
		this.moduleName = moduleName;
		this.className = className;
	}

	/** {@inheritDoc} */
	@Override
	public String toString() {
		return "GLOBAL:{module:\"" + moduleName + "\", class:\"" + className + "\"}";
	}
}
