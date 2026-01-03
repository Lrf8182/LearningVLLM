import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen3-1.7b"


print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")


messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "解释一下为什么天空是蓝色的？请用简短的一句话回答。"},
]


input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,  # True (Default): Returns a list of integers (token IDs). Use this when passing
    # data directly to a model.
    # False: Returns a raw string. Use this if you need to debug the formatting or if you are using
    #  an inference server that handles tokenization separately (like vLLM or TGI).
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)


print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


print("Generating...")
outputs = model.generate(input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)


generated_ids = outputs[0][len(input_ids[0]) :]
response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("-" * 50)
print(f"User Input: {messages[1]['content']}")
print(f"AI Response: {response_text}")
print("-" * 50)
