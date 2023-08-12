package org.rogmann.llm.demo;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

import org.rogmann.llm.LlmConfigException;
import org.rogmann.llm.LlmExecutor;
import org.rogmann.llm.LlmWorkerPoolPhaser;
import org.rogmann.llm.ModelReader;
import org.rogmann.llm.bloom.BloomModel;
import org.rogmann.llm.tokenizer.BPETokenizer;
import org.rogmann.llm.tokenizer.Tokenizer;

/**
 * Executes a text generation using a BLOOM based model.
 */
public class DemoSimpleMain {

	/**
	 * Entry method.
	 * @param args model-folder
	 */
	public static void main(String[] args) throws IOException, LlmConfigException {
		if (args.length == 0) {
			throw new IllegalArgumentException("Usage: model-folder");
		}
		final File folder = new File(args[0]);
		final Tokenizer tokenizer = new BPETokenizer(folder);
		
		final ModelReader modelReader = new ModelReader(folder, true);
		final int nThreads = 8;

		try (LlmExecutor executor = new LlmWorkerPoolPhaser(nThreads)) {
			final BloomModel model = new BloomModel(modelReader, 1, executor);
	
			//String inputSentence = "Translate to Chinese: I write a program in Java.";
			String inputSentence = "What is the capital of France?";
			//String inputSentence = "Translate to chinese: cat.";
			//String inputSentence = "¿Quién era Joan Miró?";
			final int maxToken = 10;
			final List<String> listToken = model.computeNextTokens(tokenizer, model, inputSentence, maxToken);
			System.out.println("Prompt: " + inputSentence);
			final String sResult = listToken.stream().collect(Collectors.joining());
			System.out.println("Result:" + sResult);
		}
	}
}
