package org.rogmann.llm.pickle;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llm.LlmConfigException;

public class PickleReader {
	/** logger */
	private static final Logger LOG = Logger.getLogger(PickleReader.class.getName());
	
	/** version of pickle-stream */
	private final int pickleVersion;

	private final Map<String, Object> memo = new HashMap<>();
	
	/** current position in pickle-stream */
	private int pos;

	/** Finished object */
	private final Object result;

	public PickleReader(InputStream is, PickleReducer reducer) throws IOException, LlmConfigException {
		final PickleOpcode opVersion = readOpcode(is);
		if (opVersion != PickleOpcode.PROTO) {
			throw new IOException("Unexpected opcode (expected PROTO): " + opVersion);
		}
		pickleVersion = readUInt8(is);
		LOG.fine("pickle-version: " + pickleVersion);

		Stack<Object> stack = new Stack<>();
		while (true) {
			final PickleOpcode op = readOpcode(is);
			if (op == PickleOpcode.STOP) {
				break;
			}
			if (LOG.isLoggable(Level.FINEST)) {
				LOG.finest(String.format("0x%04x: Opcode=%s, stack.size=%d",
						Integer.valueOf(pos), op, Integer.valueOf(stack.size())));
			}
			try {
				switch (op) {
				case MARK:
					stack.push(PickleMark.OBJECT);
					break;
				case GLOBAL:
				{
					final String moduleName	= readStringN(is);
					final String className	= readStringN(is);
					stack.push(new PickleGlobal(moduleName, className));
					break;
				}
				case BINPUT:
				case LONG_BINPUT:
				{
					final int idx = (op == PickleOpcode.BINPUT) ? readUInt8(is) : readInt32LE(is);
					final Object obj = stack.peek();
					memo.put(Integer.toString(idx), obj);
					if (LOG.isLoggable(Level.FINEST)) {
						LOG.finest(String.format("%s: memo[%d] = %s", op, Integer.valueOf(idx), obj));
					}
					break;
				}
				case BINGET:
				case LONG_BINGET:
				{
					final int idx = (op == PickleOpcode.BINGET) ? readUInt8(is) : readInt32LE(is);
					final Object obj = memo.get(Integer.toString(idx));
					if (LOG.isLoggable(Level.FINEST)) {
						LOG.finest(String.format("%s: memo[%d] = %s", op, Integer.valueOf(idx), obj));
					}
					stack.push(obj);
					break;
				}
				case BININT:
				{
					final int val = readInt32LE(is);
					stack.push(Integer.valueOf(val));
					if (LOG.isLoggable(Level.FINEST)) {
						LOG.finest(String.format("%s: %d", op, Integer.valueOf(val)));
					}
					break;
				}
				case BININT1:
				{
					final int val = readUInt8(is);
					stack.push(Integer.valueOf(val));
					break;
				}
				case BININT2:
				{
					final int val = readUInt16LE(is);
					stack.push(Integer.valueOf(val));
					break;
				}
				case BINUNICODE:
				{
					final int len = readInt32LE(is);
					final String s = readString(is, len);
					stack.push(s);
					break;
				}
				case NEWFALSE:
					stack.push(Boolean.FALSE);
					break;
				case EMPTY_DICT:
					stack.push(new HashMap<String, Object>());
					break;
				case EMPTY_TUPLE:
					stack.push(new Object[2]);
					break;
				case TUPLE:
				{
					int idxMark = getMarkObjectIndex(stack, op);
					final int size = stack.size() - 1 - idxMark;
					final Object[] tuple = new Object[size];
					for (int j = 0; j < size; j++) {
						tuple[j] = stack.get(idxMark + 1 + j);
					}
					for (int j = stack.size() - 1; j >= idxMark; j--) {
						stack.remove(j);
					}
					stack.push(tuple);
					break;
				}
				case TUPLE1:
				{
					int size = stack.size();
					if (size < 1) {
						throw new IllegalStateException(String.format("stack size %d at 0x%x: %s",
								Integer.valueOf(size), Integer.valueOf(pos), op));
					}
					final Object[] tuple = new Object[1];
					tuple[0] = stack.pop();
					stack.push(tuple);
					break;
				}
				case TUPLE2:
				{
					int size = stack.size();
					if (size < 2) {
						throw new IllegalStateException(String.format("stack size %d at 0x%x: %s",
								Integer.valueOf(size), Integer.valueOf(pos), op));
					}
					final Object[] tuple = new Object[2];
					tuple[1] = stack.pop();
					tuple[0] = stack.pop();
					stack.push(tuple);
					break;
				}
				case SETITEM:
				{
					int size = stack.size();
					if (size < 3) {
						throw new IllegalStateException(String.format("stack size %d at 0x%x: %s",
								Integer.valueOf(size), Integer.valueOf(pos), op));
					}
					@SuppressWarnings("unchecked")
					final Map<String, Object> mapDict = (Map<String, Object>) stack.get(size - 3);
					final Object oValue = stack.pop();
					final String key = (String) stack.pop();
					mapDict.put(key, oValue);
					if (LOG.isLoggable(Level.FINEST)) {
						LOG.finest(String.format("Added key %s in dict with value of type %s",
								key, (oValue != null) ? oValue.getClass().getName() : null));
					}
					break;
				}
				case SETITEMS:
				{
					final int idxMark = getMarkObjectIndex(stack, op);
					@SuppressWarnings("unchecked")
					final Map<String, Object> mapDict = (Map<String, Object>) stack.get(idxMark - 1);
					if ((stack.size() - 1 - idxMark) % 2 != 0) {
						throw new IllegalStateException(String.format("Uneven number %d of elements after mark at %d for %s in stack of size %d at 0x%x",
								Integer.valueOf(stack.size() - 1 - idxMark), Integer.valueOf(idxMark), op,
								Integer.valueOf(stack.size()), Integer.valueOf(pos)));
					}
					final int numPairs = (stack.size() - 1 - idxMark) / 2;
					for (int i = 0; i < numPairs; i++) {
						final Object oValue = stack.pop();
						final String key = (String) stack.pop();
						mapDict.put(key, oValue);
					}
					stack.pop(); // remove mark
					if (LOG.isLoggable(Level.FINEST)) {
						LOG.finest(String.format("Added %d %s in dict with keys %s",
								Integer.valueOf(numPairs), (numPairs == 1) ? "item" : "items",
								mapDict.keySet()));
					}
					break;
				}
				case BINPERSID:
				{
					checkStack(stack, op);
					final Object oPersistantId = stack.pop();
					if (!(oPersistantId instanceof Object[])) {
						throw new IllegalStateException(String.format("Unexpected non-array PID at 0x%x: %s",
								Integer.valueOf(pos), oPersistantId));
					}
					final Object[] aPid = (Object[]) oPersistantId;
					if (aPid.length < 5) {
						throw new IllegalStateException(String.format("PID.length<5 at 0x%x: %s",
								Integer.valueOf(pos), Arrays.toString(aPid)));
					}
					if ("storage".equals(aPid[0]) && aPid[1] instanceof PickleGlobal) {
						final Storage storage = new Storage((PickleGlobal) aPid[1], (String) aPid[2], (String) aPid[3], ((Integer) aPid[4]).intValue());
						if (LOG.isLoggable(Level.FINER)) {
							LOG.finer("Storage: " + storage);
						}
						stack.push(storage);
					}
					else {
						throw new IllegalStateException(String.format("Unexpected PID-type (%s) at 0x%x: %s",
								aPid[0], Integer.valueOf(pos), Arrays.toString(aPid)));
					}
					break;
				}
				case REDUCE:
				{
					final Object[] args = (Object[]) stack.pop();
					final Object obj = stack.pop();
					Object reducedObject = reducer.reduce(obj, args);
					if (Void.TYPE.equals(reducedObject)) {
							reducedObject = null;
					} else if (reducedObject == null) {
						throw new IllegalStateException(String.format("Can't reduce object %s at 0x%x: %s",
								obj, Integer.valueOf(pos)));
					}
					stack.push(reducedObject);
					break;
				}
				case BUILD:
				{
					final Object args = stack.pop();
					final Object obj = stack.pop();
					Object builtObject = reducer.build(obj, args);
					if (builtObject == null) {
						throw new IllegalStateException(String.format("Can't built object %s at 0x%x: %s",
								obj, Integer.valueOf(pos)));
					}
					stack.push(builtObject);
					break;
				}
				default:
					throw new IllegalStateException("Unsupported opcode " + op);
				}
			}
			catch (RuntimeException e) {
				throw new IllegalStateException(String.format("An error occured while processing opcode %s at 0x%d with stack of size %d",
						op, Integer.valueOf(pos), Integer.valueOf(stack.size())), e);
			}
		}
		if (LOG.isLoggable(Level.FINE)) {
			LOG.fine(String.format("End of pickle-stream at 0x%x, stack.size=%d",
					Integer.valueOf(pos), Integer.valueOf(stack.size())));
		}
		if (stack.size() == 0) {
			throw new IllegalStateException("Stack is empty after executiong pickle-opcodes.");
		}
		final Object objFinished = stack.peek();
		result = objFinished;
	}

	/**
	 * Gets the result.
	 * @param clazz expected class of the result
	 * @param <T> type of the result
	 * @return result
	 * @throws LlmConfigException in case of an unexpected class
	 */
	public <T> T getResult(Class<T> clazz) throws LlmConfigException {
		if (!clazz.isInstance(result)) {
			throw new LlmConfigException(String.format("Result-object of type %s is not compatible with %s",
					(result != null) ? result.getClass() : null, clazz));
		}
		@SuppressWarnings("unchecked")
		final T t = (T) result;
		return t;
	}

	/**
	 * Gets the index of the top-most mark-object on the stack.
	 * @param stack stack
	 * @param op current opcode
	 * @return index of mark-object
	 */
	private int getMarkObjectIndex(Stack<Object> stack, final PickleOpcode op) {
		int i = stack.size() - 1;
		while (i >= 0 && PickleMark.OBJECT != stack.get(i)) {
			i--;
		}
		if (i < 0) {
			throw new IllegalStateException(String.format("no mark object on stack %s at 0x%x for %s",
					stack, Integer.valueOf(pos), op));
		}
		return i;
	}

	private void checkStack(Stack<Object> stack, final PickleOpcode op) {
		if (stack.isEmpty()) {
			throw new IllegalStateException(String.format("Empty stack at 0x%x: %s",
					Integer.valueOf(pos), op));
		}
	}

	/**
	 * Reads the next opcode.
	 * @param is input-stream
	 * @return opcode
	 * @throws IOException in case of an IO-error
	 * @throws IllegalStateException in case of an unknown opcode
	 */
	private PickleOpcode readOpcode(InputStream is) throws IOException {
		final int b = is.read();
		if (b == -1) {
			throw new IOException("Unexpected end of stream at " + pos);
		}
		PickleOpcode op = PickleOpcode.lookup((byte) b);
		if (op == null) {
			throw new IllegalStateException(String.format("Unknown opcode 0x%02x at 0x%x",
					Integer.valueOf(b & 0xff), Integer.valueOf(pos)));
		}
		pos++;
		return op;
	}

	private String readStringN(InputStream is) throws IOException {
		try (ByteArrayOutputStream baos = new ByteArrayOutputStream(20)) {
			while (true) {
				final int b = is.read();
				if (b == -1) {
					throw new IOException("Unexpected end of stream at " + pos);
				}
				pos++;
				if (b == '\n') {
					break;
				}
				baos.write(b);
			}
			return new String(baos.toByteArray(), StandardCharsets.UTF_8);
		}
	}

	private String readString(InputStream is, final int len) throws IOException {
		final byte[] buf = new byte[len];
		for (int i = 0; i < len; i++) {
			final int b = is.read();
			if (b == -1) {
				throw new IOException("Unexpected end of stream at " + pos);
			}
			pos++;
			buf[i] = (byte) b;
		}
		return new String(buf, StandardCharsets.UTF_8);
	}

	/**
	 * Reads an integer (8 bit).
	 * @param is input-stream
	 * @return integer
	 * @throws IOException in case of an IO-error
	 */
	private int readUInt8(InputStream is) throws IOException {
		final int b = is.read();
		if (b == -1) {
			throw new IOException("Unexpected end of stream at " + pos);
		}
		pos++;
		return b;
	}

	private int readUInt16LE(InputStream is) throws IOException {
		int i = readUInt8(is);
		i += (readUInt8(is) << 8);
		return i;
	}

	protected int readInt32BE(InputStream is) throws IOException {
		int i = (readUInt8(is) << 24);
		i += (readUInt8(is) << 16);
		i += (readUInt8(is) << 8);
		i += readUInt8(is);
		return i;
	}

	private int readInt32LE(InputStream is) throws IOException {
		int i = readUInt8(is);
		i += (readUInt8(is) << 8);
		i += (readUInt8(is) << 16);
		i += (readUInt8(is) << 24);
		return i;
	}

}
