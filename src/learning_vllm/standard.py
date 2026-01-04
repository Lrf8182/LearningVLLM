def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. SETUP: Load Model and Tokenizer
    # We use Qwen2.5-1.5B-Instruct as the standard "small but smart" baseline.
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    # The prompt we want to test
    user_query = "Explain why the sky is blue in one sentence."

    print("\n" + "=" * 50)
    print("METHOD 1: WITH apply_chat_template (Recommended)")
    print("=" * 50)

    # Define standard message structure
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_query},
    ]

    # A. Apply template to get tensor inputs directly
    inputs_with_template = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # B. Generate
    outputs_template = model.generate(
        inputs_with_template,
        max_new_tokens=100,
        do_sample=False,  # Deterministic for testing
    )

    print("#" * 80)
    print("output_template: ")
    print("\n")
    print(outputs_template)
    print("#" * 80)

    # C. Decode (skipping the prompt tokens to show only the answer)
    response_template = tokenizer.decode(
        outputs_template[0][len(inputs_with_template[0]) :], skip_special_tokens=True
    )

    print(f"Output:\n{response_template}")

    print("\n" + "=" * 50)
    print("METHOD 2: WITHOUT apply_chat_template (Manual Formatting)")
    print("=" * 50)

    # Qwen uses the ChatML format.
    # We must manually construct the string exactly as the model expects it.
    # Format: <|im_start|>role\ncontent<|im_end|>\n
    manual_prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_query}<|im_end|>\n"
        "<|im_start|>assistant\n"  # Trigger the model to start generating here
    )

    # A. Tokenize the raw string
    inputs_manual = tokenizer(manual_prompt, return_tensors="pt").to(model.device)

    # B. Generate
    outputs_manual = model.generate(**inputs_manual, max_new_tokens=100, do_sample=False)

    # C. Decode
    response_manual = tokenizer.decode(
        outputs_manual[0][inputs_manual.input_ids.shape[1] :], skip_special_tokens=True
    )

    print(f"Output:\n{response_manual}")

    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)

    # Let's verify if our manual string matches what apply_chat_template produces internally
    template_debug_string = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Template String: {repr(template_debug_string)}")
    print(f"Manual String:   {repr(manual_prompt)}")

    if template_debug_string == manual_prompt:
        print(">> Success: The manual formatting matches the template exactly.")
    else:
        print(">> Notice: slight difference in formatting.")
