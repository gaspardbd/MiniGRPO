# Fast Training Guide with LoRA

## Problem Solved

Original `train.py` was too slow: 4 minutes for 3 samples due to:
- Training the full model (500M parameters)
- Separate reference model on CPU
- No optimization

`train_fast.py` is 10x faster through:
- LoRA: trains only ~1% of parameters
- No separate reference model
- Optimized batch sizes and rollouts
- Optional 4-bit quantization

## Comparison with TRL GRPOTrainer

| Aspect | TRL GRPOTrainer | train_fast.py |
|--------|----------------|---------------|
| LoRA | Yes | Yes |
| Quantization | 4-bit | Optional 4-bit |
| Reference model | Implicit | Uses current log probs |
| Implementation | HuggingFace TRL | Custom |
| Rewards | 2 separate functions | 1 combined function |
| Control | Limited | Full control |

## Usage

### Colab

```python
%cd /content
!git clone https://github.com/gaspardbd/MiniGRPO.git
%cd MiniGRPO
!pip install -q peft bitsandbytes

from huggingface_hub import login
login("hf_xxxxx")

%run train_fast.py
```

### Local

```bash
cd MiniGRPO
pip install peft bitsandbytes
python train_fast.py
```

## Configuration

Default settings in `train_fast.py`:

```python
# Model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
use_lora = True
use_quant = False  # Set True if memory issues

# Dataset
num_prompts = 500  # Limited from 100k
batch_size = 8     # Increased from 4
num_rollout = 2    # Reduced from 4

# Training
max_steps = 100
learning_rate = 1e-5
clip_epsilon = 0.28
kl_weight = 0.0  # Beta = 0

# LoRA
lora_r = 64
lora_alpha = 128
lora_dropout = 0.1
```

## Expected Performance

| Configuration | Time per step | 100 steps |
|--------------|---------------|-----------|
| train.py | ~9 min | ~15h |
| train_fast.py (LoRA) | ~1 min | ~1h40 |
| train_fast.py (LoRA + quant) | ~45 sec | ~1h15 |

## Checkpoints and Evaluation

### Saved Checkpoints

- `output_fast/step_25/` - Checkpoint at 25 steps
- `output_fast/step_50/` - Checkpoint at 50 steps
- `output_fast/final/` - Final LoRA adapters
- `output_fast/final_merged/` - Merged model (LoRA + base)

### Evaluation

```bash
# Full test set
python evaluate.py --model_path output_fast/final_merged

# Quick test (100 samples)
python evaluate.py --model_path output_fast/final_merged --num_samples 100

# Intermediate checkpoint
python evaluate.py --model_path output_fast/step_50
```

The evaluation script:
- Auto-detects LoRA checkpoints
- Merges adapters automatically
- Uses same prompt format as training
- Calculates accuracy on GSM8K
- Saves results to JSON

### Example Output

```
============================================================
EVALUATION RESULTS
============================================================
Accuracy: 34.52%
Correct: 453/1319
============================================================

Sample predictions:

Example 1:
  Question: Janet's ducks lay 16 eggs per day...
  True answer: 18
  Generated: 18
  Correct: Yes
```

## Customization

### Longer training

```python
num_prompts = 2000
max_steps = 500
```

### More speed

```python
use_quant = True
num_rollout = 1
batch_size = 16
max_length = 200
```

### Less memory (if OOM)

```python
use_quant = True
batch_size = 4
num_rollout = 1
max_length = 150
```

## Troubleshooting

### CUDA out of memory

```python
use_quant = True
batch_size = 2
num_rollout = 1
```

### Missing dependencies

```bash
pip install peft bitsandbytes
```

### Rewards always 0

Model doesn't generate correct format. Run:
```bash
python test_generation.py
```

### Loss explodes (>100)

```python
learning_rate = 5e-6  # Reduce from 1e-5
```

## Differences from train.py

| Feature | train.py | train_fast.py |
|---------|----------|---------------|
| LoRA | No | Yes |
| Reference model | Separate on CPU | Uses current log probs |
| Speed | 9 min/step | 1 min/step |
| Memory | High | Low |
| Dataset | 100k prompts | 500 prompts (configurable) |
| Batch size | 4 | 8 |
| Rollouts | 4 | 2 |

## Complete Workflow

```bash
# 1. Test model
python test_generation.py

# 2. Train
python train_fast.py

# 3. Quick evaluation
python evaluate.py --model_path output_fast/final_merged --num_samples 100

# 4. Full evaluation (if satisfied)
python evaluate.py --model_path output_fast/final_merged
```

## Notes

- **LoRA**: Trains only low-rank matrices (~1% of parameters)
- **No ref model**: Uses current policy log probs as reference instead of separate model
- **Quantization**: Optional 4-bit quantization reduces memory by 4x
- **num_prompts**: Default 500 for speed, increase as needed
