"""
Fast GRPO training with LoRA and quantization
Similar to TRL GRPOTrainer but custom implementation
"""

from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from grpo_loss import GRPO_Loss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch

def load_model_with_lora(model_name: str, use_lora: bool = True, use_quant: bool = True):
    """Load model with optional LoRA and quantization for speed"""
    
    # Quantization config
    quantization_config = None
    if use_quant:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = torch.float16
        print(f"Using 4-bit quantization", flush=True)
    else:
        if torch.cuda.is_available():
            try:
                supports_bf16 = torch.cuda.is_bf16_supported()
            except Exception:
                supports_bf16 = False
            dtype = torch.bfloat16 if supports_bf16 else torch.float16
        else:
            dtype = torch.float32
    
    print(f"Loading {model_name} (dtype={dtype}, quant={use_quant}, lora={use_lora})...", flush=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training if quantized
    if use_quant:
        model = prepare_model_for_kbit_training(model)
    
    model.config.use_cache = False
    
    # Apply LoRA
    if use_lora:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        print("LoRA applied:", flush=True)
        model.print_trainable_parameters()
    
    print(f"Loaded {model_name}", flush=True)
    
    # Enable gradient checkpointing (saves memory)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    return model, tokenizer

def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
    num_rollout: int,
    temperature: float,
    max_length: int,
    top_p: float,
    system_prompt: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Generate rollouts without reference model (use old log probs instead)"""
    model.eval()
    
    # Prepare chat prompt
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    
    # Tokenize
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)
    
    # Repeat for num_rollout
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(num_rollout, 1)
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(num_rollout, 1)
    
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=1.05,
        max_length=max_length,
        pad_token_id=pad_token_id,
        use_cache=True,  # Use cache during generation for speed
    )
    
    # Generate
    with torch.no_grad():
        seq_ids = model.generate(**model_inputs, generation_config=generation_config)
    
    # Decode completions
    completions = tokenizer.batch_decode(
        seq_ids[:, model_inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Create action mask
    action_mask = torch.zeros_like(seq_ids, dtype=torch.bool)
    action_mask[:, model_inputs["input_ids"].shape[1]:] = True
    action_mask[seq_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    
    # Calculate rewards
    rewards = torch.zeros(num_rollout, dtype=torch.float32)
    for i, completion in enumerate(completions):
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )
        
        pred_answer = answer_match.group(1).strip() if answer_match else None
        reward = 0.0
        if pred_answer is not None:
            if pred_answer == answer:
                reward = 1.0
            elif answer in pred_answer:
                reward = 0.5
            else:
                reward = 0.0
        rewards[i] = reward
    
    return seq_ids, action_mask.to(seq_ids.device), rewards, completions

def read_prompts(data_path, limit=None):
    """Read prompts from jsonl file with optional limit"""
    rows = []
    file_path = Path(data_path)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows

# System prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

def main():
    # ==================== PARAMETERS ====================
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    seed = 42
    
    # Training parameters (optimized for speed)
    batch_size = 8  # Increased from 4
    num_rollout = 2  # Reduced from 4 for speed
    num_step_epochs = 1
    temperature = 1.0  # Like in your TP
    top_p = 1.0
    
    # LoRA and quantization
    use_lora = True  # CRITICAL for speed
    use_quant = False  # Set to True if you have GPU memory issues
    
    # Dataset
    prompts_path = "math_tasks.jsonl"
    num_prompts = 500  # Use only 500 prompts instead of 100k for speed
    
    # Training length
    max_steps = 100
    
    # Model saving
    checkpoint_path = Path("./output_fast")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = 25
    
    # Length limits
    max_length = 300  # Reduced from 512 for speed
    
    # Optimizer
    learning_rate = 1e-5  # Like in your TP
    
    # Loss
    clip_epsilon = 0.28  # Like in your TP
    kl_weight = 0.0  # Beta = 0 like in your TP
    
    # ==================== SETUP ====================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Load model with LoRA
    model, tokenizer = load_model_with_lora(
        model_name, 
        use_lora=use_lora, 
        use_quant=use_quant
    )
    pad_token_id = tokenizer.eos_token_id
    
    # Optimizer (use paged_adamw for efficiency)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load dataset
    prompts = read_prompts(prompts_path, limit=num_prompts)
    dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=True)
    total_steps = len(dataloader)
    actual_steps = min(total_steps, max_steps)
    
    print(f"\n{'='*60}", flush=True)
    print(f"FAST TRAINING CONFIGURATION", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"LoRA: {use_lora}, Quantization: {use_quant}", flush=True)
    print(f"Loaded {len(prompts)} prompts (limited from full dataset)", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Rollouts per sample: {num_rollout}", flush=True)
    print(f"Total possible steps: {total_steps}", flush=True)
    print(f"Steps to run: {actual_steps}", flush=True)
    print(f"Checkpoint interval: every {checkpoint_interval} steps", flush=True)
    print(f"Learning rate: {learning_rate}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Loss and buffer
    loss_fn = GRPO_Loss(clip_epsilon=clip_epsilon, kl_weight=kl_weight)
    replay_buffer = ReplayBuffer()
    
    # ==================== TRAINING LOOP ====================
    for k, prompt_batch in enumerate(dataloader):
        if k >= max_steps:
            print(f"\n🛑 Reached max_steps={max_steps}. Stopping training.", flush=True)
            break
        
        replay_buffer.clear()
        progress_pct = ((k + 1) / actual_steps) * 100
        print(f"\n{'='*60}", flush=True)
        print(f"Step {k+1}/{actual_steps} ({progress_pct:.1f}%): starting rollouts", flush=True)
        print(f"{'='*60}", flush=True)
        
        question_batch = prompt_batch["question"]
        answer_batch = prompt_batch["answer"]
        all_rewards = []
        
        # ========== ROLLOUT PHASE ==========
        with torch.no_grad():
            for i, (q, a) in enumerate(zip(question_batch, answer_batch)):
                print(f"  Sample {i+1}/{len(question_batch)}: generating {num_rollout} rollouts...", flush=True)
                
                seq_ids, action_mask, rewards, completions = rollout(
                    model, tokenizer, q, a,
                    num_rollout=num_rollout,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    system_prompt=system_prompt,
                )
                all_rewards.extend(rewards.tolist())
                print(f"  Sample {i+1}: done. Rewards: {rewards.tolist()}", flush=True)
                
                # DEBUG first sample
                if k == 0 and i == 0:
                    print(f"\n  DEBUG - First completion:", flush=True)
                    print(f"  Question: {q[:80]}...", flush=True)
                    print(f"  Expected: {a}", flush=True)
                    print(f"  Generated: {completions[0][:200]}...", flush=True)
                    print(f"  ---", flush=True)
                
                # Get current policy log probs and use them as "reference"
                attention_mask = seq_ids != pad_token_id
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(~attention_mask.bool(), 1)
                
                outputs = model.forward(
                    input_ids=seq_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
                
                logits = outputs["logits"]
                log_probs = F.log_softmax(logits[:, :-1], dim=-1)
                log_probs = log_probs.gather(
                    dim=-1, index=seq_ids[:, 1:].unsqueeze(-1)
                ).squeeze(dim=-1)
                
                # Use current log probs as reference (no separate ref model)
                log_probs_ref = log_probs.detach().clone()
                
                # Calculate advantages
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                advantages = advantages.to(seq_ids.device).unsqueeze(-1)
                
                # Calculate KL
                log_ratio = log_probs_ref.float() - log_probs.float()
                if action_mask is not None:
                    log_ratio = log_ratio * action_mask
                kl = log_ratio.exp() - log_ratio - 1
                
                # Store experience
                experience = Experience(
                    sequences=seq_ids,
                    action_log_probs=log_probs.detach(),
                    log_probs_ref=log_probs_ref,
                    returns=rewards.unsqueeze(-1).to(seq_ids.device),
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))
        
        torch.cuda.empty_cache()
        
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        print(f"\nRollout Summary:", flush=True)
        print(f"  Buffer size: {len(replay_buffer)}", flush=True)
        print(f"  Mean reward: {mean_reward:.4f}", flush=True)
        print(f"  Min/Max: {min(all_rewards):.2f}/{max(all_rewards):.2f}", flush=True)
        
        # ========== TRAINING PHASE ==========
        print(f"\nTraining phase...", flush=True)
        
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )
        
        model.train()
        epoch_losses = []
        
        for epoch in range(num_step_epochs):
            batch_losses = []
            for batch_idx, experience in enumerate(experience_sampler):
                experience = experience.to(device)
                attention_mask = experience.attention_mask
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(~attention_mask.bool(), 1)
                
                outputs = model.forward(
                    input_ids=experience.sequences,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                
                log_probs = F.log_softmax(outputs["logits"][:, :-1], dim=-1)
                log_probs = log_probs.gather(
                    dim=-1, index=experience.sequences[:, 1:].unsqueeze(-1)
                ).squeeze(dim=-1)
                
                loss_value = loss_fn(log_probs, experience)
                
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                
                batch_losses.append(loss_value.item())
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}: loss={loss_value.item():.4f}", flush=True)
            
            epoch_mean_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            epoch_losses.append(epoch_mean_loss)
        
        overall_mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        print(f"\nStep {k+1}/{actual_steps} completed!", flush=True)
        print(f"  Mean loss: {overall_mean_loss:.4f}", flush=True)
        print(f"  Mean reward: {mean_reward:.4f}", flush=True)
        
        # Save checkpoint
        if (k + 1) % checkpoint_interval == 0:
            checkpoint_dir = checkpoint_path / f"step_{k+1}"
            print(f"\nSaving checkpoint to {checkpoint_dir}...", flush=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved!", flush=True)
    
    # Save final model
    final_dir = checkpoint_path / "final"
    print(f"\nSaving final model to {final_dir}...", flush=True)
    if use_lora:
        # Save LoRA adapters
        model.save_pretrained(final_dir)
        # Also save merged model
        print("Merging LoRA adapters with base model...", flush=True)
        model = model.merge_and_unload()
        merged_dir = checkpoint_path / "final_merged"
        model.save_pretrained(merged_dir)
        print(f"Merged model saved to {merged_dir}", flush=True)
    else:
        model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Training completed!", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()

