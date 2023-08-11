package org.rogmann.llm.nn;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llm.LlmConfigException;
import org.rogmann.llm.LlmExecutor;
import org.rogmann.llm.pickle.PickleGlobal;
import org.rogmann.llm.pickle.PickleReaderBase;
import org.rogmann.llm.pickle.Storage;

/**
 * An implementation of a pickle-reducer which knows some torch-objects.
 */
public class PickleReducerTorch extends PickleReaderBase {
	/** logger */
	private static final Logger LOGGER = Logger.getLogger(PickleReducerTorch.class.getName());

	/** LLM-executor */
	private final LlmExecutor executor;

	/**
	 * Constructor
	 * @param executor LLM-executor
	 */
	public PickleReducerTorch(final LlmExecutor executor) {
		this.executor = executor;
	}

	/** {@inheritDoc} */
	@Override
	public Object reduce(Object obj, Object[] args) throws LlmConfigException {
		Object reducedObject = super.reduce(obj, args);
		if (reducedObject == null && (obj instanceof PickleGlobal)) {
			final PickleGlobal global = (PickleGlobal) obj;
			if ("torch._utils".equals(global.moduleName) && "_rebuild_tensor_v2".equals(global.className)) {
				if (args.length != 6) {
					throw new IllegalArgumentException(String.format("Unexpected number %d of args for %s: %s",
							Integer.valueOf(args.length), global.className, Arrays.toString(args)));
				}
				final Storage storage = (Storage) args[0];
				final int storageOffset = ((Integer) args[1]).intValue();
				final Object[] oSize = (Object[]) args[2];
				final int[] size = new int[oSize.length];
				for (int i = 0; i < oSize.length; i++) {
					size[i] = ((Integer) oSize[i]).intValue();
				}
				final Object[] oStride = (Object[]) args[3];
				final int[] stride = new int[oStride.length];
				for (int i = 0; i < oStride.length; i++) {
					stride[i] = ((Integer) oStride[i]).intValue();
				}
				final boolean requiresGrad = ((Boolean) args[4]).booleanValue();
				if (LOGGER.isLoggable(Level.FINE)) {
					LOGGER.fine(String.format("Build tensor %s with dimensions %s and stride %s, requiresGrad=%s",
							storage, Arrays.toString(size), Arrays.toString(stride), Boolean.toString(requiresGrad)));
				}
				reducedObject = new Tensor(storage, storageOffset, size, stride, requiresGrad, executor);
			}
		}
		return reducedObject;
	}

	/** {@inheritDoc} */
	@Override
	public Object build(Object obj, Object args) {
		Object objBuilt = super.build(obj, args);
		if (objBuilt != null) {
			return objBuilt;
		}
		throw new IllegalArgumentException("Unexpected object " + obj + " with " + args);
	}

}

