### TEST 1 - Disable prefix caching

import os
from vllm import LLM, SamplingParams

# :white_check_mark: FORCE SINGLE CORE / SINGLE PROCESS
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROCESSING_GPU_ID"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"


def print_outputs(outputs, engine):
  print("-" * 50)
  for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
    print("-" * 50)

  # Print cache-related metrics
  for m in engine.llm_engine.get_metrics():
    if "cache" in m.name:
      print(m.name, m.value)


def main():
  # :white_check_mark: Granite model (as required)
  MODEL = "ibm-granite/granite-3.3-8b-instruct"

  # Sampling parameters
  sampling_params = SamplingParams(
    max_tokens=20
  )

  # Prompts
  prompts = [
    "What are IBMs main businesses?",
  ]

  # :white_check_mark: LLM INIT
  engine = LLM(
    model=MODEL,
    enable_prefix_caching=True,
    enforce_eager=False,
    dtype="float16",
    disable_log_stats=False,
    max_model_len=1024,
    tensor_parallel_size=1,  # :white_check_mark: TP=2 (as required)
    model_impl="transformers",
  )

  # :white_check_mark: Run inference
  outputs = engine.generate(prompts[0], sampling_params)

  # :white_check_mark: Print results
  print_outputs(outputs, engine)


if __name__ == "__main__":
  main()
