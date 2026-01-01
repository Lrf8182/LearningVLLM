import torch
from enum import Enum
from typing import Union,List, Dict,Optional,Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

class EngineType(Enum):
    HF="hf"
    VLLM="vllm"

class TinyLLM:
    def __init__(
        self,
        engine_type: EngineType,
        engine_params: Dict[str, Any],
        tokenizer_params: Dict[str, Any],
        sampling_params: Dict[str,Any]
    ):
        self.engine_type=engine_type
        self.sampling_params=sampling_params
        # Listen to tokenizer_params first
        t_model= tokenizer_params.get("pretrained_model_name_or_path", engine_params.get("pretrained_model_name_or_path"))
        # For simplicity, we use the model_name from engine_params if tokenizer_name isn't specific.
        if not t_model:
            # Fallback if specific key isn't standard
            t_model=list(engine_params.values())[0]

        print(f"[{self.engine_type.value.upper()}] Initializing...")

        if self.engine_type==EngineType.HF:
            self._init_hf(engine_params,tokenizer_params)
        elif self.engine_type==EngineType.VLLM:
            self._init_vllm(engine_params,tokenizer_params)
        else:
            raise ValueError("Invalid engine type")
        
    # _ means internal private function
    def _init_hf(self,engine_params:Dict,tokenizer_params:Dict):
        # 1. Load Tokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(**tokenizer_params)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token

        # 2. Load Model
        if "device_map" not in engine_params:
            engine_params["device_map"]="auto"
        self.model=AutoModelForCausalLM.from_pretrained(**engine_params)
        self.model.eval()

    def _init_vllm(self,engine_params:Dict,tokenizer_params:Dict):
        # 1. Load Tokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(**tokenizer_params)
        self.model=LLM(**engine_params)

    def generate(self,inputs:Union[str,List[str],Dict, List[Dict], List[List[Dict]]])->Union[str,List[str]]:
        is_batch=False
        is_chat=False
        final_prompts=[]

        if isinstance(inputs,str):
            final_prompts=[inputs]
            is_batch=False
            is_chat=False

        # batch Strings
        elif isinstance(inputs,list) and len(inputs)>0 and isinstance(inputs[0],str):
            final_prompts=inputs
            is_batch=True
            is_chat=False
        
        # Case 3: Single Chat (List[Dict]) or Chat Object (Dict)
        # {"q":1, "b":2} or  [{"role": "user", "content": "hi"},{}]
        elif isinstance(inputs, dict) or (isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict)):
            convo=[inputs] if isinstance(inputs,dict) else inputs
            prompt_str=self.tokenizer.apply_chat_template(convo, tokenize=False,add_generation_prompt=True)
            final_prompts=[prompt_str]
            is_batch=False
            is_chat=True

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
    
    def _generate_hf_core(self,prompts:List[str])->List[str]:
        # 1. Tokenize (batch)
        model_inputs=self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        with torch.no_grad():
            generated_ids=self.model.generate(
                **model_inputs,
                **self.sampling_params,
                pad_token_id=self.tokenizer.pad_token_id
            )
        # Get the length of the input text(number of token)   input_ids is a two-dimensional matrix (tensor), usually of shape [Batch_Size, Sequence_Length]. 
        input_len=model_inputs.input_ids.shape[1]
        new_tokens=generated_ids[:,input_len:]
        decoded_output=self.tokenizer.batch_decode(new_tokens,skip_special_tokens=True)
        return decoded_output
            
    def _generate_vllm_core(self,prompts:List[str])->List[str]:
        vllm_sampling=SamplingParams(**self.sampling_params)
        outputs=self.model.generate(prompts,vllm_sampling)
        return [output.outputs[0].text for output in outputs]
    

def run_test_suite(
    backend_name: str, 
    model_id: str, 
    # ---New configurable parameters (with defaults) ---
    temperature: float = 0.7,
    max_tokens: int = 50,
    gpu_memory_utilization: float = 0.8, # only vLLM 
    device_map: str = "auto",            # only HF 
    do_sample: bool = True               # only HF Explicit required, vLLM handles it automatically
):
    backend_name = backend_name.lower()
    print(f"\n{'='*60}\nTESTING {backend_name.upper()} BACKEND\n{'='*60}")
    print(f" Config: Temp={temperature} | MaxTokens={max_tokens} | GPU_Util={gpu_memory_utilization}")

    tokenizer_config = {
        "pretrained_model_name_or_path": model_id,
        "trust_remote_code": True
    }

    if backend_name == "hf":
        engine_type = EngineType.HF     
        engine_params = {
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": True,
            "device_map": device_map 
        }   
        sampling_params = {
            "max_new_tokens": max_tokens, # HF : max_new_tokens
            "temperature": temperature,
            "do_sample": do_sample
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
        }

        sampling_params = {
            "max_tokens": max_tokens, # vLLM 叫 max_tokens
            "temperature": temperature
        }
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


    try:
        llm = TinyLLM(
            engine_type=engine_type,
            engine_params=engine_params,
            tokenizer_params=tokenizer_config,
            sampling_params=sampling_params
        )
    except Exception as e:
        print(f" Initialization failed: {e}")
        return

    # ... (后续的测试用例代码 Generate ... 保持不变)
    print(f"Output: {llm.generate('The capital of France is')}")

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1. 正常运行 (使用默认值)
    # run_test_suite("vllm", MODEL)

    # 2. 调试：显存不够了？调低显存占用！
    # run_test_suite("vllm", MODEL, gpu_memory_utilization=0.5)

    # 3. 实验：让模型变得更有创造力 (调高 temperature)
    run_test_suite("vllm", MODEL, temperature=0.9, max_tokens=100)
    
    # 4. 调试：强制用 CPU 跑 Hugging Face
    # run_test_suite("hf", MODEL, device_map="cpu")