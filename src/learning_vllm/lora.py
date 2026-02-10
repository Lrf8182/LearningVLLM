# # Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# print("model:", model)

# import torch
# print("Model parameters:", model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of trainable parameters: {trainable_params}")
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")


from peft import LoraConfig, get_peft_model,PeftModel

# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,  # is generally recommended to set it to 2 times r. It is similar to the addition of the learning rate, which determines the "weight" of the influence of the LoRA weights on the original model
#     lora_dropout=0.1, #Randomly deactivates 10% of LoRA neurons during training to prevent overfitting.
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
# )

# model=get_peft_model(model, lora_config)

# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total number of parameters: {total_params}")
# print(f"Number of trainable parameters: {trainable_params}")
# print("model with LoRA:", model)

# text="The 2024 edition of the Olympic Men's Football Tournamenthas now concluded,with Spain taking gold at one of the world's Football was first included at the OlympicGames at Paris 1900 - and Ferenc Puskas, Lionel Messi and Neymar are among a wealth of After France hosted the tournament in 2024,FIFA lists the most successful nations in the event's rich history"

# from datasets import Dataset
# dataset = Dataset.from_dict({"text": [text]})
# def tokenize(batch):
#     out=tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
#     out["labels"] = out["input_ids"].copy()  # For language modeling, the labels are typically the same as the input IDs, as the model learns to predict the next token in the sequence.
#     return out

# train_dataset = dataset.map(tokenize, batched=True,remove_columns=["text"])  #Drops the original text column after processing, keeping only the tensors needed for training.   #MAP: Traverse each row (or batch) in the dataset and apply the functions you specify to it.


# from transformers import TrainingArguments, Trainer
# args= TrainingArguments(
#     report_to="none",  # 禁用第三方日志。不把训练数据传到 Weights & Biases 或 TensorBoard，只在本地打印
#     output_dir="z-others/lora_output",
#     gradient_accumulation_steps=4, # matters when batch size is small,  模型运行 4 次小 Batch 后再统一更新一次参数。效果等同于把 batch_size 扩大了 4 倍
#     num_train_epochs=30, # tiny dataset -> need more epochs to learn effectively
#     learning_rate=5e-4, # high LR for tiny dataset 快速收敛，且不设置学习率衰减（constant），保持学习动力。
#     lr_scheduler_type="constant", # No decay, as we have a small dataset and want to maintain a steady learning rate
#     logging_steps=1, # Log every 1 steps to monitor training progress
#     save_steps=10,
#     remove_unused_columns=False, #如果不设为 False，Trainer 会自动删掉数据集中不符合模型定义的列。当你使用自定义格式或 LoRA 时，设为 False 更安全。
# )

# trainer=Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
# )

# trainer.train()

# messages = [
#     {"role": "user", "content": "Who is the winner of 2024 world cup?"},
# ]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=40)
# # print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
# # print("inputs:", inputs)
# print("outputs:", outputs)


# # Save Lora adapter
# model.save_pretrained("z-others/lora_adapter")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
adapter_path = "z-others/lora_adapter"
trained_model = PeftModel.from_pretrained(model, adapter_path)
trained_model.eval()
merged_model = trained_model.merge_and_unload()  # 将 LoRA 权重合并到原始模型中，并卸载 LoRA 模块，得到一个完整的模型。
messages = [
    {"role": "user", "content": "Who is the winner of 2024 world cup?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(trained_model.device)

outputs = trained_model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
outputs_merged = merged_model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs_merged[0][inputs["input_ids"].shape[-1]:]))

