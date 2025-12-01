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
    

# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # SETUP: Define Parameters
    # Using a small Qwen model for demonstration
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 1. Engine Params (passed to model init)
    engine_config = {
        "pretrained_model_name_or_path": MODEL_ID,
        "trust_remote_code": True
    }
    
    # 2. Tokenizer Params
    tokenizer_config = {
        "pretrained_model_name_or_path": MODEL_ID,
        "trust_remote_code": True
    }

    # # 3. Sampling Params
    # # Note: These keys must work for both HF and vLLM or be intersection safe.
    # # HF uses 'max_new_tokens', vLLM uses 'max_tokens'. 
    # # For this demo, we use keys compatible with HF, and if running vLLM, 
    # # you might need to map them manually in _generate_vllm_core. 
    # # Here we use HF standard.
    # gen_config = {
    #     "max_new_tokens": 50,
    #     "temperature": 0.7,
    #     "do_sample": True
    # }

    # print("\n" + "="*60)
    # print("TESTING HUGGING FACE BACKEND")
    # print("="*60)

    # # Initialize HF
    # tiny_hf = TinyLLM(
    #     engine_type=EngineType.HF,
    #     engine_params=engine_config,
    #     tokenizer_params=tokenizer_config,
    #     sampling_params=gen_config
    # )

    # # Case A: Single String
    # print("\n--- Test A: Single Raw String ---")
    # res_str = tiny_hf.generate("The capital of France is")
    # print(f"Output: {res_str}")

    # # Case B: Batch String
    # print("\n--- Test B: Batch Raw Strings ---")
    # res_batch_str = tiny_hf.generate(["1+1=", "The opposite of hot is"])
    # print(f"Output: {res_batch_str}")

    # # Case C: Single Chat (List[Dict])
    # print("\n--- Test C: Single Chat ---")
    # chat_single = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Who wrote Hamlet?"}
    # ]
    # res_chat = tiny_hf.generate(chat_single)
    # print(f"Output: {res_chat}")

    # # Case D: Batch Chat (List[List[Dict]])
    # print("\n--- Test D: Batch Chat ---")
    chat_1 = [{"role": "user", "content": "Hello!"}]
    chat_2 = [{"role": "user", "content": "Goodbye!"}]
    # res_batch_chat = tiny_hf.generate([chat_1, chat_2])
    # print(f"Output: {res_batch_chat}")

    # -------------------------------------------------
    # OPTIONAL: Test vLLM (requires CUDA and installation)
    # -------------------------------------------------
    if 1:
        print("\n" + "="*60)
        print("TESTING VLLM BACKEND")
        print("="*60)
        
        # vLLM requires slightly different sampling keys usually (max_tokens vs max_new_tokens)
        # Adjusting specifically for vLLM
        vllm_gen_config = {"max_tokens": 50, "temperature": 0.7}
        # Adjusting engine config for vLLM (needs 'model' key usually)
        vllm_engine_config = {"model": MODEL_ID, "trust_remote_code": True,"gpu_memory_utilization": 0.8}

        tiny_vllm = TinyLLM(
            engine_type=EngineType.VLLM,
            engine_params=vllm_engine_config,
            tokenizer_params=tokenizer_config,
            sampling_params=vllm_gen_config
        )
        

    
        # Case A: Single String
        print("\n--- Test A: Single Raw String ---")
        res_str = tiny_vllm.generate("The capital of France is")
        print(f"Output: {res_str}")
        print("\n")
        # Case B: Batch String
        print("\n--- Test B: Batch Raw Strings ---")
        res_batch_str = tiny_vllm.generate(["1+1=", "The opposite of hot is"])
        print(f"Output: {res_batch_str}")
        print("\n")
        # Case C: Single Chat (List[Dict])
        print("\n--- Test C: Single Chat ---")
        chat_single = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who wrote Hamlet?"}
        ]
        res_chat = tiny_vllm.generate(chat_single)
        print(f"Output: {res_chat}")
        print("\n")

        print("\n--- vLLM Test: Batch Chat ---")
        print(tiny_vllm.generate([chat_1, chat_2]))
        print("\n")