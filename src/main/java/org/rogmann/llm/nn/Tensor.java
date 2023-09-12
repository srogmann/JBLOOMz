package org.rogmann.llm.nn;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.rogmann.llm.LlmConfigException;
import org.rogmann.llm.LlmExecutor;
import org.rogmann.llm.ModelReaderBinary;
import org.rogmann.llm.pickle.Storage;

/**
 * Class working on tensors.
 */
public class Tensor {
	/** Logger */
	private static final Logger LOG = Logger.getLogger(Tensor.class.getName());

	/** storage */
	protected final Storage storage;
	
	/** storage-offset */
	protected final int storageOffset;

	/** LLM-executor */
	protected final LlmExecutor executor;

	/** dimensions (size) */
	protected final int[] shape;

	/** stride array */
	private final int[] stride;

	/** <code>true</code> if tensor requires gradients */
	protected boolean requiresGrad;

	/** float data of a 1-dimensional tensor */
	public final float[] t1;

	/** float data of a 2-dimensional tensor */
	public final float[][] t2;

	/** float data of a 2-dimensional tensor */
	public final float[][][] t3;

	/** first dimension */
	final int dim1;

	/** second dimension */
	final int dim2;

	/** third dimension */
	final int dim3;

	/**
	 * Constructor
	 * @param storage storage (e.g. FloatStorage)
	 * @param storageOffset storage-offset (e.g. 0)
	 * @param size dimensions
	 * @param stride stride
	 * @param requiresGrad <code>true</code> if tensor requires gradients
	 * @param executor LLM-executor
	 * @throws LlmConfigException in case of a configuration error or out of memory error
	 */
	public Tensor(final Storage storage, final int storageOffset, final int[] size,
			final int[] stride, final boolean requiresGrad, final LlmExecutor executor) throws LlmConfigException {
		this.storage = storage;
		this.storageOffset = storageOffset;
		this.shape = size;
		this.stride = stride;
		this.requiresGrad = requiresGrad;
		this.executor = executor;
		if (size.length == 1) {
			dim1 = size[0];
			dim2 = 0;
			dim3 = 0;
			t1 = new float[dim1];
			t2 = null;
			t3 = null;
		}
		else if (size.length == 2) {
			dim1 = size[0];
			dim2 = size[1];
			dim3 = 0;
			t1 = null;
			try {
				t2 = new float[dim1][dim2];
			} catch (OutOfMemoryError e) {
				throw new LlmConfigException(String.format("Can't allocate %s-tensor %d \u00d7 %d",
						storage, Integer.valueOf(dim1), Integer.valueOf(dim2)), e);
			}
			t3 = null;
		}
		else {
			throw new LlmConfigException("Unsupported shape " + Arrays.toString(size));
		}
	}

	/**
	 * Constructor of a three-dimensional float32-tensor.
	 * @param d1 first dimension
	 * @param d2 second dimension
	 * @param d3 third dimension
	 * @param executor executor
	 */
	public Tensor(int d1, int d2, int d3, LlmExecutor executor) {
		storage = null;
		storageOffset = 0;
		this.executor = executor;
		this.shape = new int[] { d1, d2, d3 };
		this.stride = new int[] { d2 * d3, d3, 1 };
		requiresGrad = false;
		t1 = null;
		t2 = null;
		t3 = new float[d1][d2][d3];
		dim1 = d1;
		dim2 = d2;
		dim3 = d3;
	}

	/**
	 * Reads the data of a tensor.
	 * @param key name of the tensor
	 * @param readerBinary binary-reader
	 * @throws IOException in case of an IO-error
	 * @throws LlmConfigException in case of a configuration-error
	 */
	public void readTensorData(String key, ModelReaderBinary readerBinary) throws IOException, LlmConfigException {
		String entryName = "data/" + storage.key;
		StorageFormat format = StorageFormat.lookupByTorchName(storage.type.className);
		if (shape.length == 1 ) {
			if (format == StorageFormat.FLOAT16) {
				readTensorFloat16(readerBinary.getAsStream(entryName), t1, executor);
			}
			else if (format == StorageFormat.BFLOAT16) {
				readTensorBFloat16(readerBinary.getAsStream(entryName), t1, executor);
			}
			else {
				readTensorFloat32(readerBinary.getAsStream(entryName), t1, executor);
			}
		}
		else if (shape.length == 2 ) {
			if (format == StorageFormat.FLOAT16) {
				readTensorFloat16(readerBinary.getAsStream(entryName), t2, executor);
			}
			else if (format == StorageFormat.BFLOAT16) {
				readTensorBFloat16(readerBinary.getAsStream(entryName), t2, executor);
			}
			else {
				readTensorFloat32(readerBinary.getAsStream(entryName), t2, executor);
			}
		}
		else {
			throw new LlmConfigException("Unsupported shape " + Arrays.toString(shape));
		}
	}

	public static float[] readTensor(File file, int dim, LlmExecutor executor) throws IOException {
		final float[] tensor = new float[dim];
		try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file))) {
			readTensorFloat32(bis, tensor, executor);
		}
		return tensor;
	}

	public static float[][] readTensor(File file, int dim1, int dim2, LlmExecutor executor) throws IOException {
		final float[][] tensor = new float[dim1][dim2];
		try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file))) {
			readTensorFloat32(bis, tensor, executor);
		}
		catch (IOException e) {
			throw new IOException("IO-error while reading " + file.getName(), e);
		}
		return tensor;
	}

	/**
	 * Reads an array of float16-numbers.
	 * @param is input-stream
	 * @param tensor tensor to be filled
	 * @param executor executor
	 * @throws IOException in case of an IO-error
	 */
	static void readTensorFloat16(InputStream is, float[] tensor, LlmExecutor executor) throws IOException {
		final int dim1 = tensor.length;
		final byte[] buf = new byte[dim1 * 2];
		final int len = is.read(buf);
		if (len < 2 * dim1) {
			throw new IOException(String.format("Unexpected end of file (2 * dim1 = %d, len = %d)", 2 * dim1, len));
		}
		executor.startTasks((idxThread, nThreads) -> () -> {
			final int blockSize = dim1 / nThreads;
			final int iStart = idxThread * blockSize;
			final int iEnd = (idxThread + 1) * blockSize;
			int bufIdx = iStart * 2;
			for (int i = iStart; i < iEnd; i++) {
				final int val0 = buf[bufIdx + 1];
				final int val1 = buf[bufIdx];
				int e = (val0 == 0 && val1 == 0) ? 0 : ((val0 & 0x7c) >> 2) - 15 + 127;
				final int iFloat = ((val0 & 0x80) << 24)
						+ ((e & 0xff) << 23)
						+ ((val0 & 0x03) << 21)
						+ ((val1 & 0xff) << 13);
				tensor[i] = Float.intBitsToFloat(iFloat);
				bufIdx += 2;
			}
		});
	}

	/**
	 * Reads an array of bfloat16-numbers.
	 * @param is input-stream
	 * @param tensor tensor to be filled
	 * @param executor executor
	 * @throws IOException in case of an IO-error
	 */
	static void readTensorBFloat16(InputStream is, float[] tensor, LlmExecutor executor) throws IOException {
		final int dim1 = tensor.length;
		final byte[] buf = new byte[dim1 * 2];
		final int len = is.read(buf);
		if (len < 2 * dim1) {
			throw new IOException(String.format("Unexpected end of file (2 * dim1 = %d, len = %d)", 2 * dim1, len));
		}
		executor.startTasks((idxThread, nThreads) -> () -> {
			final int blockSize = dim1 / nThreads;
			final int iStart = idxThread * blockSize;
			final int iEnd = (idxThread + 1) * blockSize;
			int bufIdx = iStart * 2;
			for (int i = iStart; i < iEnd; i++) {
				final int val0 = buf[bufIdx + 1];
				final int val1 = buf[bufIdx];
				final int iFloat = ((val0 & 0xff) << 24)
						+ ((val1 & 0xff) << 16);
				tensor[i] = Float.intBitsToFloat(iFloat);
				bufIdx += 2;
			}
		});
	}

	/**
	 * Reads an array of float32-numbers.
	 * @param is input-stream
	 * @param tensor tensor to be filled
	 * @param executor executor
	 * @throws IOException in case of an IO-error
	 */
	static void readTensorFloat32(InputStream is, float[] tensor, LlmExecutor executor) throws IOException {
		final int dim1 = tensor.length;
		final byte[] buf = new byte[dim1 * 4];
		final int len = is.read(buf);
		if (len < 4 * dim1) {
			throw new IOException(String.format("Unexpected end of file (4 * dim1 = %d, len = %d)", 4 * dim1, len));
		}
		executor.startTasks((idxThread, nThreads) -> () -> {
			final int blockSize = dim1 / nThreads;
			final int iStart = idxThread * blockSize;
			final int iEnd = (idxThread + 1) * blockSize;
			int bufIdx = iStart * 4;
			for (int i = iStart; i < iEnd; i++) {
				final int iFloat = ((buf[bufIdx + 3] & 0xff) << 24)
						+ ((buf[bufIdx + 2] & 0xff) << 16)
						+ ((buf[bufIdx + 1] & 0xff) << 8)
						+ (buf[bufIdx] & 0xff);
				tensor[i] = Float.intBitsToFloat(iFloat);
				bufIdx += 4;
			}
		});
	}

	/**
	 * Reads a matrix of float16-numbers.
	 * @param is input-stream
	 * @param tensor tensor to be filled
	 * @param executor executor
	 * @throws IOException in case of an IO-error
	 */
	static void readTensorFloat16(InputStream is, float[][] tensor, LlmExecutor executor) throws IOException {
		final int dim1 = tensor.length;
		final int dim2 = tensor[0].length;
		final byte[] buf = new byte[dim1 * dim2 * 2];
		final int len = is.read(buf);
		if (len < 2 * dim1 * dim2) {
			throw new IOException(String.format("Unexpected end of file (2 * dim2 = %d, len = %d)", 2 * dim2, len));
		}
		executor.startLoopTasks(dim1, (iStart, iEnd) -> () -> {
			for (int i = iStart; i < iEnd; i++) {
				int bufIdx = i * (dim2 * 2);
				for (int j = 0; j < dim2; j++) {
					final int val0 = buf[bufIdx + 1];
					final int val1 = buf[bufIdx];
					int e = (val0 == 0 && val1 == 0) ? 0 : ((val0 & 0x7c) >> 2) - 15 + 127;
					final int iFloat = ((val0 & 0x80) << 24)
							+ ((e & 0xff) << 23)
							+ ((val0 & 0x03) << 21)
							+ ((val1 & 0xff) << 13);
					tensor[i][j] = Float.intBitsToFloat(iFloat);
					bufIdx += 2;
				}
			}
		});
	}

	/**
	 * Reads a matrix of float16-numbers.
	 * @param is input-stream
	 * @param tensor tensor to be filled
	 * @param executor executor
	 * @throws IOException in case of an IO-error
	 */
	static void readTensorBFloat16(InputStream is, float[][] tensor, LlmExecutor executor) throws IOException {
		final int dim1 = tensor.length;
		final int dim2 = tensor[0].length;
		final byte[] buf = new byte[dim1 * dim2 * 2];
		final int len = is.read(buf);
		if (len < 2 * dim1 * dim2) {
			throw new IOException(String.format("Unexpected end of file (2 * dim2 = %d, len = %d)", 2 * dim2, len));
		}
		executor.startLoopTasks(dim1, (iStart, iEnd) -> () -> {
			for (int i = iStart; i < iEnd; i++) {
				int bufIdx = i * (dim2 * 2);
				for (int j = 0; j < dim2; j++) {
					final int val0 = buf[bufIdx + 1];
					final int val1 = buf[bufIdx];
					final int iFloat = ((val0 & 0xff) << 24)
							+ ((val1 & 0xff) << 16);
					tensor[i][j] = Float.intBitsToFloat(iFloat);
					bufIdx += 2;
				}
			}
		});
	}

	/**
	 * Reads a matrix of float32-numbers.
	 * @param is input-stream
	 * @param tensor tensor to be filled
	 * @param executor executor
	 * @throws IOException in case of an IO-error
	 */
	static void readTensorFloat32(InputStream is, float[][] tensor, LlmExecutor executor) throws IOException {
		final int dim1 = tensor.length;
		final int dim2 = tensor[0].length;
		final byte[] buf = new byte[dim1 * dim2 * 4];
		final int len = is.read(buf);
		if (len < 4 * dim1 * dim2) {
			throw new IOException(String.format("Unexpected end of file (4 * dim2 = %d, len = %d)", 4 * dim2, len));
		}
		executor.startLoopTasks(dim1, (iStart, iEnd) -> () -> {
			for (int i = iStart; i < iEnd; i++) {
				int bufIdx = i * (dim2 * 4);
				for (int j = 0; j < dim2; j++) {
					final int iFloat = ((buf[bufIdx + 3] & 0xff) << 24)
							+ ((buf[bufIdx + 2] & 0xff) << 16)
							+ ((buf[bufIdx + 1] & 0xff) << 8)
							+ (buf[bufIdx] & 0xff);
					tensor[i][j] = Float.intBitsToFloat(iFloat);
					bufIdx += 4;
				}
			}
		});
	}

	/**
	 * Executes a batch matrix-matrix product:
	 * multResult = beta * input + alpha * (batch1 * batch2).
	 * batch1 and batch2 are view in fusedQkv, the view consists of heads.
	 * 
	 * input is this tensor (e.g. ALiBi-tensor) of shape executor(batchSize * numHeads, 1, numSeq)
	 * @param numSeq length of sequence
	 * @param fusedQkv tensor containing batch1 and batch2
	 * @param numBlocks number of blocks
	 * @param numHeads number of heads
	 * @param headDim dimension of a head
	 * @param idxBlock1 block-index of batch1
	 * @param idxBlock2 block-index of batch2
	 * @param alpha alpha-factor of product
	 * @param beta beta-factor of input
	 * @param output result to be computed (batchSize, numHeads, numSeq, numSeq)
	 */
	public void baddbmmView4(int numSeq, float[][][] fusedQkv, int numBlocks,
			int numHeads, int headDim,
			int idxBlock1, int idxBlock2,
			float alpha, float beta, float[][][][] output) {
		float[][][] input = t3;
		final int batchSize = output.length;
		if (LOG.isLoggable(Level.FINER)) {
			LOG.finer("fusedQkv.length = " + fusedQkv.length + ", numHeads = " + numHeads);
			LOG.finer("alpha = " + alpha + ", beta = " + beta);
			LOG.finer("numSeq = " + numSeq + ", batchSize = " + batchSize);
		}
		for (int idxB = 0; idxB < batchSize; idxB++) {
			final int b = idxB;
			executor.startLoopTasks(numHeads, (hStart, hEnd) -> () -> {
				for (int h = hStart; h < hEnd && h < numHeads; h++) {
					for (int i = 0; i < numSeq; i++) {
						for (int j = 0; j < numSeq; j++) {
							float sum = 0f;
							for (int k = 0; k < headDim; k++) {
								sum += fusedQkv[b][i][(h * 3 + idxBlock1) * headDim + k]
										* fusedQkv[b][j][(h * 3 + idxBlock2) * headDim + k];
							}
							sum *= alpha;
							try {
								sum += beta * input[b * numHeads + h][0][j];
							} catch (ArrayIndexOutOfBoundsException e) {
								throw new RuntimeException(String.format("AIOOBE: batchSize=%d, b=%d, numHeads=%d, h=%d, j=%d, input.length=%d",
										batchSize, b, numHeads, h, j, input.length), e);
							}
							output[b][h][i][j] = sum;
						}
					}
				}
			});
		}
	}

	/**
	 * Executes a batch matrix product.
	 * @param multResult batch of left matrix, shape (batchSize, numHeads, numSeq, numSeq)
	 * @param fusedQkv tensor containing right matrix in one of its blocks
	 * @param numSeq number of sequence-entries in fusedQkv to be used (number of tokens)
	 * @param numBlocks number of blocks in fusedQkv
	 * @param numHeads number of heads
	 * @param headDim dimension of a head
	 * @param idxBlock2 index of block of the right matrix
	 * @param contextLayer tensor to be filled, shape (batch_size, seq_length, num_heads * head_dim)
	 * @param executor executor
	 */
	public static void bmmView4(float[][][][] multResult, float[][][] fusedQkv, int numSeq,
			int numBlocks, int numHeads, int headDim, int idxBlock2,
			float[][][] contextLayer, LlmExecutor executor) {
		final int batchSize = multResult.length;
		for (int idxB = 0; idxB < batchSize; idxB++) {
			final int b = idxB;
			executor.startLoopTasks(numHeads, (hStart, hEnd) -> () -> {
				for (int h = hStart; h < hEnd; h++) {
					final int hh = h * headDim;
					for (int i = 0; i < numSeq; i++) {
						for (int k = 0; k < headDim; k++) {
							float sum = 0f;
							for (int j = 0; j < numSeq; j++) {
								sum += multResult[b][h][i][j]
										* fusedQkv[b][j][(h * 3 + idxBlock2) * headDim + k];
							}
							contextLayer[b][i][hh + k] = sum;
						}
					}
				}
			});
		}
	}


	/**
	 * Adds two tensors.
	 * @param input1 first tensor
	 * @param input2 second tensor
	 * @param output output tensor
	 */
	public static void add(float[][][] input1, float[][][] input2, float[][][] output) {
		final int d1 = Math.min(input1.length, input2.length);
		final int d2 = Math.min(input1[0].length, input2[0].length);
		final int d3 = Math.min(input1[0][0].length, input2[0][0].length);
		for (int i = 0; i < d1; i++) {
			float[][] m1 = input1[i];
			float[][] m2 = input2[i];
			float[][] m3 = output[i];
			for (int j = 0; j < d2; j++) {
				float[] r1 = m1[j];
				float[] r2 = m2[j];
				float[] r3 = m3[j];
				for (int k = 0; k < d3; k++) {
					r3[k] = r1[k] + r2[k];
				}
			}
		}
	}

	/** {@inheritDoc} */
	@Override
	public String toString() {
		return String.format("Tensor:{storage:%s, shape:%s, stride:%s}",
				storage,
				Arrays.toString(shape), Arrays.toString(stride));
	}

	/**
	 * Gets the shape of the tensor.
	 * @return tensor-shape
	 */
	public int[] getShape() {
		return shape;
	}

}
