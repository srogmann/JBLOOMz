package org.rogmann.llm.pickle;

/** pickle-opcode */
public enum PickleOpcode {
	/** Opcode MARK: push special markobject on stack */
	MARK('('),
	/** Opcode STOP: every pickle ends with STOP */
	STOP('.'),
	/** Opcode POP: discard topmost stack item */
	POP('0'),
	/** Opcode POP_MARK: discard stack top through topmost markobject */
	POP_MARK('1'),
	/** Opcode DUP: duplicate top stack item */
	DUP('2'),
	/** Opcode FLOAT: push float object; decimal string argument */
	FLOAT('F'),
	/** Opcode INT: push integer or bool; decimal string argument */
	INT('I'),
	/** Opcode BININT: push four-byte signed int */
	BININT('J'),
	/** Opcode BININT1: push 1-byte unsigned int */
	BININT1('K'),
	/** Opcode LONG: push long; decimal string argument */
	LONG('L'),
	/** Opcode BININT2: push 2-byte unsigned int */
	BININT2('M'),
	/** Opcode NONE: push None */
	NONE('N'),
	/** Opcode PERSID: push persistent object; id is taken from string arg */
	PERSID('P'),
	/** Opcode BINPERSID:  "       "         "  ;  "  "   "     "  stack */
	BINPERSID('Q'),
	/** Opcode REDUCE: apply callable to argtuple, both on stack */
	REDUCE('R'),
	/** Opcode STRING: push string; NL-terminated string argument */
	STRING('S'),
	/** Opcode BINSTRING: push string; counted binary string argument */
	BINSTRING('T'),
	/** Opcode SHORT_BINSTRING: "     "   ;    "      "       "      " &lt; 256 bytes */
	SHORT_BINSTRING('U'),
	/** Opcode UNICODE: push Unicode string; raw-unicode-escaped'd argument */
	UNICODE('V'),
	/** Opcode BINUNICODE:   "     "       "  ; counted UTF-8 string argument */
	BINUNICODE('X'),
	/** Opcode APPEND: append stack top to list below it */
	APPEND('a'),
	/** Opcode BUILD: call __setstate__ or __dict__.update() */
	BUILD('b'),
	/** Opcode GLOBAL: push self.find_class(modname, name); 2 string args */
	GLOBAL('c'),
	/** Opcode DICT: build a dict from stack items */
	DICT('d'),
	/** Opcode EMPTY_DICT: push empty dict */
	EMPTY_DICT('}'),
	/** Opcode APPENDS: extend list on stack by topmost stack slice */
	APPENDS('e'),
	/** Opcode GET: push item from memo on stack; index is string arg */
	GET('g'),
	/** Opcode BINGET:   "    "    "    "   "   "  ;   "    " 1-byte arg */
	BINGET('h'),
	/** Opcode INST: build &amp; push class instance */
	INST('i'),
	/** Opcode LONG_BINGET: push item from memo on stack; index is 4-byte arg */
	LONG_BINGET('j'),
	/** Opcode LIST: build list from topmost stack items */
	LIST('l'),
	/** Opcode EMPTY_LIST: push empty list */
	EMPTY_LIST(']'),
	/** Opcode OBJ: build &amp; push class instance */
	OBJ('o'),
	/** Opcode PUT: store stack top in memo; index is string arg */
	PUT('p'),
	/** Opcode BINPUT:   "     "    "   "   " ;   "    " 1-byte arg */
	BINPUT('q'),
	/** Opcode LONG_BINPUT:   "     "    "   "   " ;   "    " 4-byte arg */
	LONG_BINPUT('r'),
	/** Opcode SETITEM: add key+value pair to dict */
	SETITEM('s'),
	/** Opcode TUPLE: build tuple from topmost stack items */
	TUPLE('t'),
	/** Opcode EMPTY_TUPLE: push empty tuple */
	EMPTY_TUPLE(')'),
	/** Opcode SETITEMS: modify dict by adding topmost key+value pairs */
	SETITEMS('u'),
	/** Opcode BINFLOAT: push float; arg is 8-byte float encoding */
	BINFLOAT('G'),

	// Protocol 2

	/** Opcode PROTO: identify pickle protocol */
	PROTO(0x80),
	/** Opcode NEWOBJ: build object by applying cls.__new__ to argtuple */
	NEWOBJ(0x81),
	/** Opcode EXT1: push object from extension registry; 1-byte index */
	EXT1(0x82),
	/** Opcode EXT2: ditto, but 2-byte index */
	EXT2(0x83),
	/** Opcode EXT4: ditto, but 4-byte index */
	EXT4(0x84),
	/** Opcode TUPLE1: build 1-tuple from stack top */
	TUPLE1(0x85),
	/** Opcode TUPLE2: build 2-tuple from two topmost stack items */
	TUPLE2(0x86),
	/** Opcode TUPLE3: build 3-tuple from three topmost stack items */
	TUPLE3(0x87),
	/** Opcode NEWTRUE: push True */
	NEWTRUE(0x88),
	/** Opcode NEWFALSE: push False */
	NEWFALSE(0x89),
	/** Opcode LONG1: push long from &lt; 256 bytes */
	LONG1(0x8a),
	/** Opcode LONG4: push really big long */
	LONG4(0x8b),

	// _tuplesize2code = [EMPTY_TUPLE, TUPLE1, TUPLE2, TUPLE3]

	// Protocol 3 (Python 3.x)

	/** Opcode BINBYTES: push bytes; counted binary string argument */
	BINBYTES('B'),
	/** Opcode SHORT_BINBYTES:  "     "   ;    "      "       "      " &lt; 256 bytes */
	SHORT_BINBYTES('C'),

	// Protocol 4

	/** Opcode SHORT_BINUNICODE: push short string; UTF-8 length &lt; 256 bytes */
	SHORT_BINUNICODE(0x8c),
	/** Opcode BINUNICODE8: push very long string */
	BINUNICODE8(0x8d),
	/** Opcode BINBYTES8: push very long bytes string */
	BINBYTES8(0x8e),
	/** Opcode EMPTY_SET: push empty set on the stack */
	EMPTY_SET(0x8f),
	/** Opcode ADDITEMS: modify set by adding topmost stack items */
	ADDITEMS(0x90),
	/** Opcode FROZENSET: build frozenset from topmost stack items */
	FROZENSET(0x91),
	/** Opcode NEWOBJ_EX: like NEWOBJ but work with keyword only arguments */
	NEWOBJ_EX(0x92),
	/** Opcode STACK_GLOBAL: same as GLOBAL but using names on the stacks */
	STACK_GLOBAL(0x93),
	/** Opcode MEMOIZE: store top of the stack in memo */
	MEMOIZE(0x94),
	/** Opcode FRAME: indicate the beginning of a new frame */
	FRAME(0x95),

	// Protocol 5

	/** Opcode BYTEARRAY8: push bytearray */
	BYTEARRAY8(0x96),
	/** Opcode NEXT_BUFFER: push next out-of-band buffer */
	NEXT_BUFFER(0x97),
	/** Opcode READONLY_BUFFER: make top of stack readonly */
	READONLY_BUFFER(0x98);

	private static PickleOpcode[] OPCODES;

	private final byte opcode;
	
	static {
		OPCODES = new PickleOpcode[256];
		for (PickleOpcode po : PickleOpcode.values()) {
			OPCODES[po.opcode & 0xff] = po;
		}
	}

	PickleOpcode(char c) {
		opcode = (byte) c;
	}

	PickleOpcode(int b) {
		opcode = (byte) b;
	}

	/**
	 * Gets the byte of an opcode.
	 * @return opcode
	 */
	public byte getOpcode() {
		return opcode;
	}

	/**
	 * Gets the opcode.
	 * @param b byte of opcode
	 * @return opcode or <code>null</code>
	 */
	public static PickleOpcode lookup(final byte b) {
		return OPCODES[b & 0xff];
	}
}
