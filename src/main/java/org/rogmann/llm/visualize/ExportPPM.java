package org.rogmann.llm.visualize;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.logging.Logger;

/** Class to export model-states as a portable-bitmap. */
public class ExportPPM {
	/** logger */
	private static final Logger LOG = Logger.getLogger(ExportPPM.class.getName());

	/**
	 * Exports the fused-qkv-layers into a PPM-image.
	 * @param file output-file
	 * @param layersFusedQkv fused-qkv-layers (layers, batchsize, tokens, 3 * hiddenSize)
	 */
	public static void exportFusedQkv(File file, float[][][][] layersFusedQkv) {
		final int nLayers = layersFusedQkv.length;
		final int nTokens = layersFusedQkv[0][0].length;
		final int nHiddenSize = layersFusedQkv[0][0][0].length / 3;
		final int wx = nHiddenSize;
		final int wy = nLayers * (nTokens + 1);
		LOG.info(String.format("Write PPM-file %s: w=%d, h=%d", file, Integer.valueOf(wx), Integer.valueOf(wy)));
		
		try (OutputStream os = new BufferedOutputStream(new FileOutputStream(file))) {
			final String header = String.format("P6 %d %d 255\n",
					Integer.valueOf(wx), Integer.valueOf(wy));
			os.write(header.getBytes(StandardCharsets.US_ASCII));
			for (int i = 0; i < nLayers; i++) {
				for (int j = 0; j < nTokens; j++) {
					for (int k = 0; k < nHiddenSize; k++) {
						final float[] layerToken = layersFusedQkv[i][0][j];
						final float fq = layerToken[3 * k];
						final float fk = layerToken[3 * k + 1];
						final float fv = layerToken[3 * k + 2];
						os.write(mapToByte(fq));
						os.write(mapToByte(fk));
						os.write(mapToByte(fv));
					}
				}
				for (int k = 0; k < nHiddenSize; k++) {
					os.write(0);
					os.write(0);
					os.write(0);
				}
			}
		}
		catch (IOException e) {
			throw new RuntimeException("IO-error while writing " + file, e);
		}
	}

	/**
	 * Exports the hidden states into a PPM-image.
	 * @param file output-file
	 * @param hiddenState hidden states (layer + 1, batchSize, numSeq, hiddenSize)
	 */
	public static void exportHiddenStates(File file, float[][][][] hiddenState) {
		final int nLayers = hiddenState.length - 1;
		final int nTokens = hiddenState[0][0].length;
		final int nHiddenSize = hiddenState[0][0][0].length;
		final int wx = nHiddenSize;
		final int wy = (nLayers + 1) * (nTokens + 1);
		LOG.info(String.format("exportHiddenStates: nLayers=%d, nTokens=%d, nHiddenSize=%d",
				Integer.valueOf(nLayers), Integer.valueOf(nTokens), Integer.valueOf(nHiddenSize)));
		LOG.info(String.format("Write PPM-file %s: w=%d, h=%d", file, Integer.valueOf(wx), Integer.valueOf(wy)));
		
		try (OutputStream os = new BufferedOutputStream(new FileOutputStream(file))) {
			final String header = String.format("P6 %d %d 255\n",
					Integer.valueOf(wx), Integer.valueOf(wy));
			os.write(header.getBytes(StandardCharsets.US_ASCII));
			for (int i = 0; i <= nLayers; i++) {
				for (int j = 0; j < nTokens; j++) {
					final float[] layerToken = hiddenState[i][0][j];
					if (layerToken == null) {
						throw new IllegalArgumentException(String.format("Missing layer-values at layer %d, token %d",
								Integer.valueOf(i), Integer.valueOf(j)));
					}
					for (int k = 0; k < nHiddenSize; k++) {
						final float value = layerToken[k];
						os.write(mapToByte(value));
						os.write(mapToByte(value));
						os.write(mapToByte(value));
					}
				}
				for (int k = 0; k < nHiddenSize; k++) {
					os.write(0);
					os.write(0);
					os.write(0);
				}
			}
		}
		catch (IOException e) {
			throw new RuntimeException("IO-error while writing " + file, e);
		}
	}

	static int mapToByte(final float f) {
		final double d = Math.atan(f);
		final double v = ((d / Math.PI) + 0.5) * 255.0;
		return (int) v;
	}

}
