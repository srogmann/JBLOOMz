# JBLOOMz

The original idea for JBLOOMz was to consider the possibility of running a large language model in a pure Java environment without python and the need of consuming REST APIs. It was clear that this approach would not be as fast and as comprehensive as pytorch, or more precisely, [🤗 Transformers](https://huggingface.co/docs/transformers/index). I you want to play with a fast and versatile framework use the python- and rust-based 🤗 Transformers.

JBLOOMz (Java-BLOOMz) is a small (<5000 loc) Java implementation of the tokenizer and model used by the multilingual language model [BLOOM](https://huggingface.co/bigscience/bloom).

For example you can download the model [bloom/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m) (see below), it has 560 million parameters and needs 4.3 GB of disk space, and use JBLOOMz to generate text based on this model.

    Prompt: Translate to Chinese: I write a program in Java.
    Response: 我在Java中写程序。</s>

    Prompt: ¿Quién era Joan Miró?
    Result: pintor español</s>El artista plástico catalán Joan Miró, conocido como el artista catalán

Sample code (see src/test/java):

    		final File folder = new File(args[0]);
    		final Tokenizer tokenizer = new BPETokenizer(folder);
    		
    		final ModelReader modelReader = new ModelReader(folder, true);
    		final int nThreads = 8;
    
    		try (LlmExecutor executor = new LlmWorkerPoolPhaser(nThreads)) {
    			final BloomModel model = new BloomModel(modelReader, 1, executor);
	    
    			String inputSentence = "Translate to Chinese: I write a program in Java.";
    			final int maxToken = 10;
    			final List<String> listToken = model.computeNextTokens(tokenizer, model, inputSentence, maxToken);
    			System.out.println("Prompt: " + inputSentence);
    			final String sResult = listToken.stream().collect(Collectors.joining());
    			System.out.println("Result:" + sResult);
    		}

It is interesting to watch the execution of matrix operations resulting in natural language.

    #tokens: 250680
    [2023-08-10 22:15:48] [INFORMATION] Start thread-pool of 8 threads 
    [2023-08-10 22:15:48] [INFORMATION] Read bloom-model 'bloomz-560m' with 24 layers, 16 heads and hidden size 1024 
    [2023-08-10 22:15:48] [INFORMATION] Read model-file pytorch_model.bin 
    [2023-08-10 22:15:49] [INFORMATION] pickle-map.size: 294 
    [2023-08-10 22:15:50] [INFORMATION] Total memory: 8304,0 MB 
    [2023-08-10 22:15:50] [INFORMATION] Free memory: 4497,3 MB 
    [2023-08-10 22:15:50] [INFORMATION] Load Layer 0 
    [...]
    [2023-08-10 22:15:50] [INFORMATION] Load Layer 22 
    [2023-08-10 22:15:50] [INFORMATION] Load Layer 23 
    
    Inference 1
    input_ids: [121447, 3453, 19814, 175921, 34]
    Start: 2023-08-10T22:15:50.140211
    [2023-08-10 22:15:50] [INFORMATION] Switched to softmax minus max: h=13, max=26,3 
    idx=0, max=368,835968, token=<unk> (<unk>)
    idx=1, max=374,651459, token=<s> (<s>)
    idx=2, max=395,473297, token=</s> (</s>)
    idx=366, max=395,569977, token= la (Ġla)
    idx=447, max=399,754517, token= un (Ġun)
    idx=34400, max=400,835480, token= artista (Ġartista)
    idx=90364, max=400,886292, token= pintor (Ġpintor)
    End: 2023-08-10T22:15:51.136294
    Token:  pintor

You can also infer fine-trained BLOOM based models, see [llm-instruction-sample](https://github.com/srogmann/llm-instruction-sample).

## Why BLOOM?

There are a lot of LLMs: GPT2, Llama, Llama2, BLOOM, MPT, RedPajama, to name a few. BLOOM contains a lot of languages and is open-access. I came across BLOOM when I looked at the models created by [Malte Ostendorff](https://ostendorff.org), e.g. <https://huggingface.co/malteos/bloom-1b5-clp-german>.

## Contents

The program has the following packages (based at org.rogmann).

* llm: This package contains classes to read a model and to execute computations in several threads.
* llm.bloom: BLOOM specific model implementation with ALiBi, attention heads, GELU and MLP.
* llm.json: A tiny JSON parser.
* llm.nn: Mathematical operations, e.g. layer-norm, softmax and matrix-multiplications.
* llm.pickle: A partial implementation of the pickle virtual machine of python.
* llm.tokenizer: a BPE (byte pair) tokenizer.

## Computations

JBLOOMz uses float only. It might read models containing FLOAT16 or BFLOAT16 but executes them using FLOAT32. Therefore very large models need a lot of memory (heap space). You need

    -Xmx32000m

to execute a 1B5 model.

## Performance

The computation uses the following interface to distribute the work on several threads:

    public interface LlmTaskLoopFunction {
    
    	/**
    	 * Create a runnable to execute a part of a loop.
    	 * @param idxStart first index
    	 * @param idxEnd last index (exclusive)
    	 * @return runnable to be executed in a worker-thread
    	 */
    	Runnable apply(int idxStart, int idxEnd);    
    }

The interface LlmExecutor is used to start a computation:

    	/**
    	 * Starts a task consisting of a loop to be executed in parallel.
    	 * @param taskFunction task-loop-function
    	 */
    	void startLoopTasks(int n, LlmTaskLoopFunction taskFunction);

The GELU (Gaussian Error Linear Unit) uses this functional interface:

    			final float[][] m1 = input[i];
    			final float[][] m2 = output[i];
    			executor.startLoopTasks(d2, (jStart, jEnd) -> () -> {
    				for (int j = jStart; j < jEnd; j++) {
    					final float[] r1 = m1[j];
    					final float[] r2 = m2[j];
    					for (int k = 0; k < d3; k++) {
    						final float x = r1[k];
    						final float z = 1f + 0.044715f * x * x;
    						final float y = 1.0f + (float) (Math.tanh(0.79788456f * x * z));
    						r2[k] = x * 0.5f * y;
    						if (Float.isNaN(r2[k])) {
    							throw new IllegalStateException("Nan");
    						}
    					}
    				}
    			});

There are different implementations of LlmExecutor:

* LlmExecutorSingleThread: It uses one thread only, definitely without race conditions.
* LlmWorkerPoolPhaser: Synchronizing the threads using phasers.
* LlmWorkerPoolReentrantLock: Synchronizing the threads using reentrant locks.
* LlmWorkerPoolBusySpin: A very CPU-intensive executor without JVM-based locking.

The multi-threaded executors are faster than the single-threaded one. But even CPU-based pytorch is seven times faster. But there is [JEP 448](https://openjdk.org/jeps/448), the vector API! I haven't tried that yet.

One question is how the different threads treat the float-arrays. I'm used to AtomicInteger and AtomicLong. But using millions of volatile floats? This implementation uses pure float\[\]\[\]\[\] so I can't guarantee that there are not race conditions reading floats when JIT optimizes the execution of the threads.

A consolation is the loading of the model at the beginning which is fast.

## Getting a model

You can get a lot of interesting [models](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) at [Hugging Face](https://huggingface.co/). You can use git or write a tiny python script to download a model, see [Downloading models](https://huggingface.co/).

JBLOOMz supports BLOOM-based models only.

On a Linux system you may install git-lfs to download a model:

    git clone --depth 1 -v https://huggingface.co/bigscience/bloomz-560m/

## Logging

JBLOOMz uses JRE-based java.util.logging only. One might use a bridge to ones favorite log-implementation.

## Further observations

### Rust ###

The [tokenizer](https://github.com/huggingface/tokenizers/) used by Hugging Face is written in [Rust](https://www.rust-lang.org/). As an example have a look at the [byte-level implementation](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs).

### pickle-VM of python

Python uses a little virtual machine to serialize and deserialize python objects on disk. It's called pickle. The class PickleReader understands the opcodes used in the serialization of a BLOOM model.

### 64-bit ZIPs

The pickle file and the weights are store in a zip-archive. Most Java 8 JREs are not able to read 64-bit zip-archives! But Java 11, Java 17, ... can read those zip-archives. To run a model in Java 8 (I know that this is a rather old runtime) one can unpack the .bin-file:

    		final boolean supportUnzippedModel = true;
    		final ModelReader modelReader = new ModelReader(folder, supportUnzippedModel);

## Support

I wrote this project in my free time and I like my free time so support is given by studying the following links:

* <https://huggingface.co/bigscience/bloom>
* BLOOM paper: <https://arxiv.org/abs/2211.05100>
* Attention Is All You Need: <https://arxiv.org/abs/1706.03762>
* <https://huggingface.co/docs/transformers/index>
* <https://github.com/huggingface/transformers/tree/main/src/transformers/models/bloom>
