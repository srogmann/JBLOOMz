package org.rogmann.llm;

import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Class to read binary data of the model (weights, ...).
 */
public class ModelReaderBinary implements Closeable {
	/** logger */
	private static final Logger LOG = Logger.getLogger(ModelReaderBinary.class.getName());

	/** zip-file containing the binary data (zipped model) */
	private final ZipFile zip;

	/** folder containing the binary data-files (unzipped model) */
	private final File folderArchive;
	
	/** prefix of an entry, e.g. "archive/" or "pytorch_model/" */
	private final String entryPrefix;

	/**
	 * Constructor
	 * @param file file containing binary data (weights, ...)
	 * @param preferUpacked <code>true</code> if the model-zip might have been unzipped on disk
	 * @throws IOException in case of an IO-error
	 */
	public ModelReaderBinary(final File file, final boolean preferUnpacked) throws IOException {
		folderArchive = preferUnpacked ? checkForUnzippedModelData(file.getParentFile()) : null;
		if (folderArchive != null) {
			zip = null;
			entryPrefix = null;
		}
		else {
			zip = new ZipFile(file);
			
			final ZipEntry firstEntry = zip.entries().nextElement();
			entryPrefix = firstEntry.getName().replaceFirst("([^/]+/).*", "$1");
		}
	}

	/**
	 * Name of the entry
	 * @param entry name, e.g. "data.pkl" or "data/103"
	 * @throws IOException in case of an IO-error
	 */
	public InputStream getAsStream(final String entry) throws IOException {
		if (folderArchive != null) {
			final File file = new File(folderArchive, entry);
			try {
				return new FileInputStream(file);
			} catch (FileNotFoundException e) {
				throw new IOException("Can't open " + file, e);
			}
		}
		final String entryName = entryPrefix + entry;
		final ZipEntry zipEntry = zip.getEntry(entryName);
		if (zipEntry == null) {
			throw new IOException("No entry " + entryName + " in " + zip.getName());
		}
		return zip.getInputStream(zipEntry);
	}

	@Override
	public void close() throws IOException {
		if (zip != null) {
			zip.close();
		}
	}

	/**
	 * Looks for a folder which contains unzipped model-data.
	 * @param parentFile base folder
	 * @return folder or <code>null</code>
	 */
	static File checkForUnzippedModelData(File folder) throws IOException {
		final File[] filesParent = folder.listFiles();
		if (filesParent == null) {
			throw new IOException("Missing model-folder " + folder);
		}
		for (File lFile : filesParent) {
			if (lFile.isDirectory()) {
				final File dirData = lFile;
				File[] filesDir = dirData.listFiles();
				if (filesDir == null) {
					continue;
				}
				for (File file : filesDir) {
					if (file.isFile() && file.getName().endsWith(".pkl")) {
						LOG.info("checkForUnzippedModelData: " + file);
						return dirData;
					}
				}
			}
		}
		return null;
	}

}
