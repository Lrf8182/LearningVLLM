# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams


from transformers import AutoTokenizer

# from learning_vllm.utils.common import test1
from .utils.common import test1  # . means current directory: directory where contains basic.py
# in a import b, in b import a will cause circular import error


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# def main():

#     llm = LLM(model="Qwen/Qwen3-1.7b", gpu_memory_utilization=0.8,max_model_len=10000)

#     outputs = llm.generate(prompts, sampling_params)

#     print("\nGenerated Outputs:\n" + "-" * 60)
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt:    {prompt!r}")
#         print(f"Output:    {generated_text!r}")
#         print("-" * 60)


# With apply chat template
def main():
    llm = LLM(model="Qwen/Qwen3-1.7b", gpu_memory_utilization=0.8, max_model_len=10000)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7b", trust_remote_code=True)
    conversations = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, my name is"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "The president of the United States is"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "The future of AI is"},
        ],
    ]
    prompts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True,  # Because there is <|im_end|>, add generation prompt must
            # be added regardless of whether it is a supplementary sentence or not.
        )
        for convo in conversations
    ]

    print(f"\n[Debug] Prompt 1 sent to vLLM:\n{prompts[0]!r}\n")

    sampling_params = SamplingParams(temperature=0.7, max_tokens=1000)
    outputs = llm.generate(prompts, sampling_params)
    print("-" * 50)
    for output in outputs:
        print(f"Generated: {output.outputs[0].text}")
        print("-" * 50)
        print("\n")


if __name__ == "__main__":
    main()
