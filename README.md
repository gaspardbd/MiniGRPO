# MiniGRPO

Custom implementation of GRPO (Group Relative Policy Optimization) for training language models on mathematical reasoning tasks.

## Fast Training with LoRA

Use `train_fast.py` for 10x faster training:

```bash
python train_fast.py  # ~1 min/step vs ~9 min/step
```

Key features:
- LoRA: trains only 1% of parameters
- Optional 4-bit quantization
- Works on smaller GPUs
- Similar results to full fine-tuning

See [FAST_TRAINING_GUIDE.md](FAST_TRAINING_GUIDE.md) for details.

## Quick Start

### Test model format first

```bash
python test_generation.py
```

This verifies the model generates `<answer>` tags correctly.

### Google Colab

```python
# Clone repository
%cd /content
!git clone https://github.com/gaspardbd/MiniGRPO.git
%cd MiniGRPO

# Install dependencies
!pip install -q peft bitsandbytes transformers datasets

# Login to HuggingFace
from huggingface_hub import login
login("hf_xxxxx")

# Run training
%run train_fast.py

# Evaluate
!python evaluate.py --model_path output_fast/final_merged --num_samples 100
```

### Local Training

```bash
pip install -r requirements_fast.txt
python train_fast.py
```

## Training Logs

During training, you'll see:
- Step progress with percentage
- Mean reward per step (0.0 to 1.0)
- Loss per batch and epoch
- Individual rewards per sample
- Checkpoint confirmations

Example output:

```
============================================================
Step 1/100 (1.0%): starting rollouts
============================================================
  Sample 1/8: generating 2 rollouts...
  Sample 1: done. Rewards: [0.5, 1.0]
  
Rollout Summary:
  Buffer size: 16
  Mean reward: 0.6875
  Min/Max: 0.00/1.00

Training phase...
  Epoch 1, Batch 1: loss=0.4523

Step 1/100 completed!
  Mean loss: 0.4207
  Mean reward: 0.6875
```

## Evaluation

Evaluate trained models on GSM8K test set:

```bash
# Full test set (1,319 samples)
python evaluate.py --model_path output_fast/final_merged

# Quick evaluation (100 samples)
python evaluate.py --model_path output_fast/final_merged --num_samples 100

# Evaluate specific checkpoint
python evaluate.py --model_path output_fast/step_50
```

The script automatically detects LoRA checkpoints and merges adapters.

Expected accuracy on GSM8K:
- Base Qwen2.5-0.5B: 10-15%
- After 100 GRPO steps: 30-35%
- After 500 GRPO steps: 35-40%

## Project Structure

### Training Scripts
- `train_fast.py` - Fast training with LoRA (recommended)
- `train.py` - Original full model training (slower)
- `test_generation.py` - Test model format

### Evaluation
- `evaluate.py` - GSM8K test set evaluation

### Core Components
- `grpo_loss.py` - GRPO loss implementation
- `replay_buffer.py` - Experience replay buffer
- `math_tasks.jsonl` - Training dataset

### Documentation
- `FAST_TRAINING_GUIDE.md` - Detailed guide
- `README.md` - This file

## Configuration

### train_fast.py (recommended)

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
use_lora = True
use_quant = False  # Set True if memory issues
batch_size = 8
num_rollout = 2
num_prompts = 500
max_steps = 100
learning_rate = 1e-5
temperature = 1.0
clip_epsilon = 0.28
kl_weight = 0.0
```

### train.py (slower, full model)

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
batch_size = 4
num_rollout = 4
temperature = 0.3
max_steps = 100
```

## Supported Models

Models that follow the `<answer>` format:
- Qwen/Qwen2.5-0.5B-Instruct (recommended)
- Qwen/Qwen2.5-1.5B-Instruct
- HuggingFaceTB/SmolLM2-1.7B-Instruct

Models that don't work:
- LiquidAI/LFM2-350M (doesn't generate `<answer>` tags)

## Troubleshooting

### CUDA out of memory
```python
# In train_fast.py
use_quant = True
batch_size = 4
num_rollout = 1
```

### Rewards always 0
Run `python test_generation.py` to verify the model generates `<answer>` tags.

### Loss explodes (>100)
```python
learning_rate = 5e-6  # Reduce from 1e-5
```

## Performance Comparison

| Metric | train.py | train_fast.py |
|--------|----------|---------------|
| Time per step | ~9 min | ~1 min |
| Memory | ~12 GB | ~3-4 GB |
| Parameters trained | 500M (100%) | ~5M (1%) |
| 100 steps | ~15 hours | ~1.7 hours |

## References

- https://github.com/mingyin0312/RLFromScratch
- https://github.com/open-thought/tiny-grpo
- https://github.com/huggingface/trl
