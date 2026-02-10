import torch
from enum import Enum
from typing import Union, List, Dict, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import math
import torch.nn.functional as F

import os
import csv


top_x = 2  # Number of top candidates to display


class EngineType(Enum):
    HF = "hf"
    VLLM = "vllm"


class TinyLLM:
    def __init__(
        self,
        engine_type: EngineType,
        engine_params: Dict[str, Any],
        tokenizer_params: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ):
        self.engine_type = engine_type
        self.sampling_params = sampling_params
        # Listen to tokenizer_params first
        # t_model = tokenizer_params.get(
        #     "pretrained_model_name_or_path",
        #     engine_params.get("pretrained_model_name_or_path"),
        # )

        # if not t_model:
        #     # Fallback if specific key isn't standard
        #     t_model = list(engine_params.values())[0]

        print(f"[{self.engine_type.value.upper()}] Initializing...")

        if self.engine_type == EngineType.HF:
            self._init_hf(engine_params, tokenizer_params)
        elif self.engine_type == EngineType.VLLM:
            self._init_vllm(engine_params, tokenizer_params)
        else:
            raise ValueError("Invalid engine type")

    # _ means internal private function
    def _init_hf(self, engine_params: Dict, tokenizer_params: Dict):
        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_params)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Model
        if "device_map" not in engine_params:
            engine_params["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(**engine_params)
        self.model.eval()

    def _init_vllm(self, engine_params: Dict, tokenizer_params: Dict):
        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_params)
        self.model = LLM(**engine_params)

    def generate(
        self, inputs: Union[str, List[str], Dict, List[Dict], List[List[Dict]]]
    ) -> Union[str, List[str]]:
        is_batch = False
        is_chat = False
        final_prompts = []

        if isinstance(inputs, str):
            final_prompts = [inputs]
            is_batch = False
            is_chat = False

        # batch Strings
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str):
            final_prompts = inputs
            is_batch = True
            is_chat = False

        # Case 3: Single Chat (List[Dict]) or Chat Object (Dict)
        # {"q":1, "b":2} or  [{"role": "user", "content": "hi"},{}]
        elif isinstance(inputs, dict) or (
            isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict)
        ):
            convo = [inputs] if isinstance(inputs, dict) else inputs
            prompt_str = self.tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True
            )
            final_prompts = [prompt_str]
            is_batch = False
            is_chat = True

        # Case 4:
        # [[{"role": "user", "content": "hi"},{"user":"q",}],[],[]]
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], list):
            # Apply template to every conversation in the batch
            final_prompts = [
                self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in inputs
            ]
            is_batch = True
            is_chat = True
        else:
            raise ValueError("Empty input or unsupported type format.")

        if self.engine_type == EngineType.HF:
            results = self._generate_hf_core(final_prompts)
        else:
            results = self._generate_vllm_core(final_prompts)

        # --- POST-PROCESSING: Return Type ---
        if is_batch:
            return results
        else:
            return results[0]

    def _generate_hf_core(self, prompts: List[str]) -> List[str]:
        # 1. Tokenize (batch)
        model_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                **self.sampling_params,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # Get the length of the input text(number of token)   input_ids is a two-dimensional matrix
        #  (tensor), usually of shape [Batch_Size, Sequence_Length].
        generated_ids = outputs.sequences
        input_len = model_inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, input_len:]
        decoded_output = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        batch_size = len(prompts)
        for b in range(batch_size):
            print(f"\n[HF Decode] Sentence {b} Text: {decoded_output[b]}")

            # 遍历该句子生成的每一个 Step
            for step_idx, step_logits in enumerate(outputs.scores):
                # step_logits 的 shape 是 [batch_size, vocab_size]
                # 获取当前句子 (b) 的 logits
                probs = F.softmax(step_logits[b], dim=-1)
                top_probs, top_indices = torch.topk(probs, k=top_x, dim=-1)

                print(f"  Token {step_idx} top {top_x} candidates:")
                for i in range(top_x):
                    t_id = top_indices[i].item()
                    t_str = self.tokenizer.decode(t_id)
                    p_val = top_probs[i].item()
                    print(f"    - '{t_str}': {p_val:.4f}")

        for b in range(batch_size):
            csv_rows = []
            # outputs.scores 包含了生成的每一步
            for step_logits in outputs.scores:
                # step_logits shape: [batch_size, vocab_size]
                probs = F.softmax(step_logits[b], dim=-1)
                top_probs, top_indices = torch.topk(probs, k=top_x, dim=-1)

                row = []
                for i in range(top_x):
                    t_id = top_indices[i].item()
                    t_str = self.tokenizer.decode(t_id)
                    p_val = top_probs[i].item()
                    row.extend([t_str, f"{p_val:.4f}"])
                csv_rows.append(row)

        return decoded_output

    def _generate_vllm_core(self, prompts: List[str]) -> List[str]:
        vllm_sampling = SamplingParams(**self.sampling_params)
        outputs = self.model.generate(prompts, vllm_sampling)

        for output in outputs:
            generated_text = output.outputs[0].text
            token_logprobs = output.outputs[0].logprobs

            print(f"Generated text: {generated_text}")
            for i, logprob_dict in enumerate(token_logprobs):
                print(f"Token {i} top {top_x} candidates:")
                print(logprob_dict)
                for token_id, logprob_obj in logprob_dict.items():
                    # Convert logprob to linear probability
                    prob = math.exp(logprob_obj.logprob)
                    print(f"  Token: {logprob_obj.decoded_token} | Prob: {prob:.4f}")

        final_texts = []
        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            final_texts.append(generated_text)
            token_logprobs = output.outputs[0].logprobs

            csv_rows = []
            if token_logprobs:
                for logprob_dict in token_logprobs:
                    row = []
                    # vLLM 的 logprobs 已经是按概率排序好的字典
                    for token_id, logprob_obj in logprob_dict.items():
                        prob = math.exp(logprob_obj.logprob)
                        row.extend([logprob_obj.decoded_token, f"{prob:.4f}"])
                    csv_rows.append(row)

            # 保存为 CSV (文件名包含 index 以区分 batch)
            self.save_probs_to_csv(csv_rows, f"vllm_output_prompt_{idx}.csv")

        return [output.outputs[0].text for output in outputs]

    def save_probs_to_csv(self, data: List[List[Any]], filename: str):
        """
        将提取的概率数据保存为 CSV
        data 格式: [[t1, p1, t2, p2...], [t1, p1, ...]] (n_tokens 行)
        """
        x = len(data[0]) // 2
        headers = []
        for i in range(1, x + 1):
            headers.extend([f"Token_{i}", f"Prob_{i}"])

        with open(filename, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)
        print(f"Successfully saved probabilities to {filename}")


def run_test_suite(
    backend_name: str,
    model_id: str,
    # ---New configurable parameters (with defaults) ---
    temperature: float = 0.7,
    max_tokens: int = 50,
    gpu_memory_utilization: float = 0.8,  # only vLLM
    device_map: str = "auto",  # only HF
    do_sample: bool = True,  # only HF Explicit required, vLLM handles it automatically
):
    backend_name = backend_name.lower()
    print(f"\n{'=' * 60}\nTESTING {backend_name.upper()} BACKEND\n{'=' * 60}")
    print(
        f" Config: Temp={temperature} | MaxTokens={max_tokens} | GPU_Util={gpu_memory_utilization}"
    )

    tokenizer_config = {
        "pretrained_model_name_or_path": model_id,
        "trust_remote_code": True,
    }

    if backend_name == "hf":
        engine_type = EngineType.HF
        engine_params = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": True,
            "device_map": device_map,
        }
        sampling_params = {
            "max_new_tokens": max_tokens,  # HF : max_new_tokens
            "temperature": temperature,
            "do_sample": do_sample,
        }

    elif backend_name == "vllm":
        try:
            import vllm
        except ImportError:
            print(" vLLM not installed.")
            return

        engine_type = EngineType.VLLM

        engine_params = {
            "model": model_id,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_logprobs": top_x,
        }

        sampling_params = {
            "max_tokens": max_tokens,  # vLLM 叫 max_tokens
            "temperature": temperature,
            "logprobs": top_x,
        }
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    try:
        llm = TinyLLM(
            engine_type=engine_type,
            engine_params=engine_params,
            tokenizer_params=tokenizer_config,
            sampling_params=sampling_params,
        )
    except Exception as e:
        print(f" Initialization failed: {e}")
        return

    # ... (后续的测试用例代码 Generate ... 保持不变)
    # print(f"Output: {llm.generate('The capital of France is')}")
    print(
        f"Output: {llm.generate(['1+1=', 'The opposite of hot is', 'What is the capital of Germany?'])}"
    )


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1. 正常运行 (使用默认值)
    # run_test_suite("vllm", MODEL)

    # 2. 调试：显存不够了？调低显存占用！
    # run_test_suite("vllm", MODEL, gpu_memory_utilization=0.5)

    print("\n" + "=" * 60 + "\n")
    print(" test vllm")
    run_test_suite("vllm", MODEL, temperature=0.9, max_tokens=10)
    print("\n" + "=" * 60 + "\n")
    print(" test hf")
    run_test_suite("hf", MODEL, temperature=0.9, max_tokens=10)

    # 4. 调试：强制用 CPU 跑 Hugging Face
    # run_test_suite("hf", MODEL, device_map="cpu")
