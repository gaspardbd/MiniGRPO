"""
Evaluation script for GRPO trained models on GSM8K test set
Based on your TP evaluation code
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import json
from datetime import datetime
import re
from pathlib import Path

# System prompts (same as training)
R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."

def extract_hash_answer(text: str) -> str | None:
    """Extract answer from GSM8K format (after ####)"""
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None

def extract_xml_answer(text: str) -> str:
    """Extract answer from <answer> tags"""
    match = re.search(r"<answer>\s*(-?\d+)\s*</answer>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def evaluate_model(
    model_path: str,
    batch_size: int = 1,
    num_samples: int = None,
    save_results: bool = True,
    use_one_shot: bool = True,
) -> Dict:
    """
    Evaluate a model on GSM8K test set
    
    Args:
        model_path: Path to the model (can be base model or LoRA checkpoint)
        batch_size: Batch size for evaluation (keep at 1 for simplicity)
        num_samples: Number of samples to evaluate (None = all)
        save_results: Whether to save results to JSON
        use_one_shot: Whether to use one-shot example (like in training)
    """
    print("="*60)
    print("GSM8K EVALUATION")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"One-shot: {use_one_shot}")
    print("="*60)
    
    # Load model and tokenizer
    print("\n📦 Loading model components...")
    
    # Detect if it's a LoRA checkpoint
    model_path_obj = Path(model_path)
    is_lora = (model_path_obj / "adapter_config.json").exists()
    
    if is_lora:
        print("Detected LoRA checkpoint. Loading with PEFT...")
        from peft import PeftModel
        
        # Load base model
        base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Adjust if different
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("✓ LoRA adapters merged with base model")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded on {model.device}")
    
    # Load test dataset
    print("\n📚 Loading GSM8K test dataset...")
    dataset = load_dataset('openai/gsm8k', 'main', split='test')
    if num_samples:
        dataset = dataset.select(range(num_samples))
    total_samples = len(dataset)
    print(f"✓ Loaded {total_samples} samples")
    
    # Evaluation loop
    results = []
    correct = 0
    total = 0
    
    progress_bar = tqdm(
        total=total_samples,
        desc="Evaluating",
        unit="examples",
        dynamic_ncols=True,
    )
    
    progress_bar.set_postfix({
        'acc': '0.00%',
        'correct': '0',
    })
    
    # Process in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        if isinstance(batch_data['question'], str):
            # Single sample
            batch_data = {
                'question': [batch_data['question']],
                'answer': [batch_data['answer']]
            }
        
        current_batch_size = len(batch_data['question'])
        
        # Prepare prompts
        prompts = []
        for q in batch_data['question']:
            conversation = [
                {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS}
            ]
            
            if use_one_shot:
                conversation.extend([
                    {'role': 'user', 'content': "What is 2+2?"},
                    {'role': 'assistant', 'content': "To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.\n<answer>4</answer>"}
                ])
            
            conversation.append({'role': 'user', 'content': q.strip()})
            prompts.append(conversation)
        
        # Convert to chat format and generate
        outputs = []
        for prompt in prompts:
            formatted_prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                return_tensors='pt',
                add_generation_prompt=True
            )
            
            with torch.no_grad():
                output = model.generate(
                    formatted_prompt.to(model.device),
                    max_new_tokens=512,
                    temperature=1.0,
                    do_sample=False,  # Greedy for evaluation
                    pad_token_id=tokenizer.eos_token_id,
                )
            outputs.append(output)
        
        # Process responses
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the assistant's response (after last user message)
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1]
            elif "assistant\n" in response:
                response = response.split("assistant\n")[-1]
            
            # Extract answers
            generated_answer = extract_xml_answer(response)
            true_answer = extract_hash_answer(batch_data['answer'][j])
            
            is_correct = generated_answer == true_answer
            
            # Store result
            result = {
                'question': batch_data['question'][j],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response[:500],  # Truncate for storage
                'correct': is_correct
            }
            results.append(result)
            
            # Update metrics
            if is_correct:
                correct += 1
            total += 1
        
        # Update progress
        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })
    
    progress_bar.close()
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat(),
        'one_shot': use_one_shot,
    }
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Correct: {correct}/{total}")
    print("="*60)
    
    # Show some examples
    print("\n📝 Sample predictions:")
    for i, result in enumerate(results[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Question: {result['question'][:80]}...")
        print(f"  True answer: {result['true_answer']}")
        print(f"  Generated: {result['generated_answer']}")
        print(f"  Correct: {'✓' if result['correct'] else '✗'}")
    
    # Save results
    if save_results:
        save_path = f"gsm8k_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(save_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'results': results
            }, f, indent=2)
        print(f"\n💾 Results saved to {save_path}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GRPO model on GSM8K")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (can be LoRA or merged model)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all test set)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--no_one_shot",
        action="store_true",
        help="Don't use one-shot example"
    )
    
    args = parser.parse_args()
    
    print("Starting GSM8K evaluation...")
    evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        use_one_shot=not args.no_one_shot,
    )

