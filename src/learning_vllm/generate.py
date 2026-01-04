# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from transformers import AutoTokenizer


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="Qwen/Qwen3-1.7b")
    # parser.set_defaults(gpu_memory_utilization=0.7)
    # parser.set_defaults(max_model_len=8192)
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


# Apply_chat_template
# tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7b",trust_remote_code=True)
# conversations = [
#     [
#         {"role": "user", "content": "Hello, my name is"}
#     ],
#     [
#         {"role": "user", "content": "The president of the United States is"}
#     ],        [
#         {"role": "user", "content": "What is the capital of France?"}
#     ],        [
#         {"role": "user", "content": "The future of AI is"}
#     ]
# ]
# prompts=[
#     tokenizer.apply_chat_template(
#         convo,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     for convo in conversations
# ]

# print(f"\n[Debug] Prompt 1 sent to vLLM:\n{prompts[0]!r}\n")

# sampling_params = SamplingParams(temperature=0.7, max_tokens=1000)
# outputs = llm.generate(prompts, sampling_params)
# print("-" * 50)
# for output in outputs:
#     print(f"Generated: {output.outputs[0].text}")
#     print("-" * 50)
#     print("\n")


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
