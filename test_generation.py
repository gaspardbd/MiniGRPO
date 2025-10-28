#!/usr/bin/env python3
"""
Quick diagnostic script to test model generation without full training.
Run this to see what the model is actually generating.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# System prompt from train.py
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

# Test question
test_question = "What is 2 + 3?"
test_answer = "5"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("🔍 Testing model generation...\n")
print(f"Model: {MODEL_NAME}")
print(f"Question: {test_question}")
print(f"Expected answer: {test_answer}\n")

# Load model
print("Loading model...")
if torch.cuda.is_available():
    try:
        supports_bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        supports_bf16 = False
    dtype = torch.bfloat16 if supports_bf16 else torch.float16
else:
    dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded on {model.device}\n")

# Prepare prompt
chat_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": test_question},
]

chat_prompt = tokenizer.apply_chat_template(
    chat_messages, 
    tokenize=False, 
    add_generation_prompt=True
)

print("=" * 70)
print("FULL PROMPT SENT TO MODEL:")
print("=" * 70)
print(chat_prompt)
print("=" * 70)

# Tokenize
model_inputs = tokenizer(
    [chat_prompt],
    return_tensors="pt",
    padding=True,
    return_attention_mask=True,
).to(model.device)

# Generate
print("\n🚀 Generating 3 completions...\n")
generation_config = GenerationConfig(
    do_sample=True,
    top_p=1.0,
    temperature=0.3,
    repetition_penalty=1.05,
    max_length=512,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

for i in range(3):
    print(f"\n{'='*70}")
    print(f"COMPLETION {i+1}:")
    print(f"{'='*70}")
    
    with torch.no_grad():
        seq_ids = model.generate(**model_inputs, generation_config=generation_config)
    
    full_output = tokenizer.decode(seq_ids[0], skip_special_tokens=True)
    completion = tokenizer.decode(
        seq_ids[0, model_inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    print(completion)
    
    # Check for answer tag
    import re
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
    
    if answer_match:
        pred_answer = answer_match.group(1).strip()
        print(f"\n✅ Found <answer> tag!")
        print(f"   Predicted: '{pred_answer}'")
        print(f"   Expected: '{test_answer}'")
        print(f"   Match: {pred_answer == test_answer}")
    else:
        print(f"\n❌ NO <answer> tag found in output!")
        print(f"   The model is not following the expected format.")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
print("""
If you see NO <answer> tags:
  → The model doesn't understand the prompt format
  → Try a different model (e.g., Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct)
  → Or adapt the prompt format to what this model expects

If you see <answer> tags but wrong answers:
  → The model is too small/weak for math reasoning
  → Need more training or a bigger/better model

If you see <answer> tags with correct answers:
  → Good! The training should work
  → Low initial rewards are normal, they'll improve
""")

