package org.rogmann.llm.pickle;

import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llm.LlmConfigException;

/**
 * Base implementation of a pickle-reducer.
 */
public class PickleReaderBase implements PickleReducer {
	/** logger */
	private static final Logger LOG = Logger.getLogger(PickleReaderBase.class.getName());

	/** {@inheritDoc} */
	@Override
	public Object reduce(Object obj, Object[] args) throws LlmConfigException {
		Object reducedObject = null;
		if ((obj instanceof PickleGlobal)) {
			final PickleGlobal global = (PickleGlobal) obj;
			if ("collections".equals(global.moduleName) && "OrderedDict".equals(global.className)) {
				reducedObject = new TreeMap<String, Object>();
			}
		}
		return reducedObject;
	}

	/** {@inheritDoc} */
	@Override
	public Object build(Object obj, Object args) {
		if (obj instanceof Map) {
			@SuppressWarnings("unchecked")
			final Map<String, Object> map = (Map<String, Object>) obj;
			if (!(args instanceof Map)) {
				throw new IllegalStateException(String.format("build: Unexpected argument %s to update a map",
						(args != null) ? args.getClass() : null));
			}
			@SuppressWarnings("unchecked")
			final Map<String, Object> mapSource = (Map<String, Object>) args;
			if (LOG.isLoggable(Level.FINEST)) {
				LOG.finest(String.format("build: add update map of size %d with map-argument of size %d",
						Integer.valueOf(map.size()), Integer.valueOf(mapSource.size())));
			}
			map.putAll(mapSource);
			return map;
		}
		return null;
	}
}
