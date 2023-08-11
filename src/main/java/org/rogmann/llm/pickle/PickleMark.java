package org.rogmann.llm.pickle;

/**
 * mark object to be placed on the stack.
 */
class PickleMark {

	/** special mark object */
	static final PickleMark OBJECT = new PickleMark();

	private PickleMark() {
		// empty constructor
	}

}
