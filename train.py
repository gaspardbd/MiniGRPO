# https://github.com/mingyin0312/RLFromScratch/blob/main/grpo_train_from_scratch.py
# https://github.com/open-thought/tiny-grpo/blob/main/train.py
# https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

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
    LlamaForCausalLM,
    GenerationConfig,
    AutoModelForCausalLM,
)
from grpo_loss import GRPO_Loss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch

def load_model(model_name: str, device_map: str):
    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    ref_model: LlamaForCausalLM,
    question: str,
    answer: str,
    num_rollout: int,
    temperature: float,
    max_length: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    #input_ids
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    chat_prompt=tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollout, 1
    )
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(num_rollout, 1)
    model_inputs = model_inputs.to(model.device)
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    # seq_ids
    seq_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        seq_ids[:, model_inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(seq_ids, dtype=torch.bool)
    action_mask[:, model_inputs["input_ids"].shape[1] :] = True
    action_mask[seq_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    #rewards
    rewards = torch.zeros(num_rollout, dtype=torch.float32)
    for i, completion in enumerate(completions):
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        pred_answer = answer_match.group(1) if answer_match else None
        reward = 0
        if pred_answer is not None:
            if pred_answer == answer:
                reward = 1.0
            elif answer in pred_answer:
                reward = 0.5
            else:
                reward = 0.0
        rewards[i] = reward
    return seq_ids, action_mask.to(seq_ids.device), rewards, completions

            

def read_prompts(data_path):
    rows=[]
    file_path = Path(data_path)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

def main():
    # parameters
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    ref_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device_map = "auto"
    seed = 42
    batch_size = 16
    max_length = 1024
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    num_step_epochs=2
    num_rollout=8
    temperature=0.6
    top_p=1.0
    prompts_path="math_tasks.jsonl"


    device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    ref_model, _ = load_model(ref_model_name, device_map="auto")
    model, tokenizer = load_model(model_name, device_map="auto")
    ref_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    pad_token_id = tokenizer.eos_token_id
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4)


    prompts = read_prompts(prompts_path)
    dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=True)

    loss_fn = GRPO_Loss(clip_epsilon=0.2, kl_weight=0.01)
    replay_buffer = ReplayBuffer()

    for k, prompt_batch in enumerate(dataloader):
        replay_buffer.clear()
        
        question_batch = prompt_batch["question"]
        answer_batch = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(question_batch, answer_batch):
                seq_ids, action_mask, rewards, completions = rollout(
                    model,
                    tokenizer,
                    ref_model,
                    q,
                    a,
                    num_rollout=num_rollout,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                attention_mask = seq_ids != pad_token_id
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(~attention_mask.bool(), 1) # RoPE in LLaMa doesnt support negative position ids
                outputs=model.forward(input_ids=seq_ids, attention_mask=attention_mask, position_ids=position_ids)
                outputs_ref=ref_model.forward(input_ids=seq_ids, attention_mask=attention_mask, position_ids=position_ids)

                logits=outputs["logits"]
                logits_ref=outputs_ref["logits"]

                log_probs=F.log_softmax(logits, dim=-1)
                log_probs=log_probs.gather(dim=-1, index=seq_ids[:, 1:].unsqueeze(-1)).squeeze(dim=-1)

                log_probs_ref=F.log_softmax(logits_ref, dim=-1)
                log_probs_ref=log_probs_ref.gather(dim=-1, index=seq_ids[:, 1:].unsqueeze(-1)).squeeze(dim=-1)

                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                advantages = advantages.unsqueeze(-1)  # [B, 1] for broadcasting over time

                log_ratio = log_probs_ref.float() - log_probs.float()
                if action_mask is not None:
                    log_ratio = log_ratio * action_mask
                kl = log_ratio.exp() - log_ratio - 1

                experience = Experience(
                    sequences=seq_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=rewards.unsqueeze(-1),
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))
            torch.cuda.empty_cache()

            experience_sampler = DataLoader(
                replay_buffer,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=join_experience_batch,
            )

            for epoch in range(num_step_epochs):
                for experience in experience_sampler:
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
                    # torch.cuda.empty_cache()
            if (
                checkpoint_path is not None
                and checkpoint_interval is not None
                and (k + 1) % checkpoint_interval == 0
            ):
                model.save_pretrained(checkpoint_path / f"step_{k}")
        if checkpoint_path is not None:
            model.save_pretrained(checkpoint_path / f"step_{k}")


                


    

if __name__ == "__main__":
    main()