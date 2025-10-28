# MiniGRPO

Re-implementation of GRPO (Group Relative Policy Optimization) from DeepSeekMath and DeepSeekR1.

## ⚡ NEW: Fast Training with LoRA

**10x faster training!** Use `train_fast.py` instead of `train.py`:

```bash
python train_fast.py  # ~1 min/step vs ~9 min/step
```

Features:
- ✅ **LoRA** - Train only 1% of parameters
- ✅ **10x faster** - 1 min/step instead of 9 min/step
- ✅ **Less memory** - Works on smaller GPUs
- ✅ **Same quality** - Similar results to full training

See [FAST_TRAINING_GUIDE.md](FAST_TRAINING_GUIDE.md) for details.

## 🚀 Quick Start

### Step 0: Test Model First (Recommended)

Before starting full training, test that the model generates the correct format:

```bash
python test_generation.py
```

This will show you 3 sample completions and verify the model uses `<answer>` tags.

### Option 1: Google Colab (Recommended)

Open `colab_train.ipynb` in Google Colab and follow the instructions. The notebook includes:
- Automated setup and dependency installation
- Real-time training logs with progress tracking
- Two methods for running training (Python or Bash)

### Option 2: Local Training

```bash
# Install dependencies
pip install torch transformers accelerate wandb

# Run training
python train.py
```

## 📊 Training Logs

During training, you'll see detailed logs including:

- **Progress**: `Step X/Y (Z%)` - Current step and completion percentage
- **Mean Reward**: Average reward across rollouts (0.0 to 1.0)
  - `1.0`: Perfect answer match
  - `0.5`: Partial answer match
  - `0.0`: No match
- **Loss**: Training loss per batch and epoch
- **Rewards**: Individual rewards for each generated completion
- **Checkpoints**: Model saves every 20 steps to `./output/`

### Example Output

```
============================================================
Step 1/25000 (0.0%): starting rollouts
============================================================
  Sample 1/4: generating 4 rollouts...
  Sample 1: generation done. Rewards: [1.0, 0.5, 0.0, 1.0]

📊 Rollout Summary:
  Buffer size: 4
  Mean reward: 0.6250
  Min/Max reward: 0.00/1.00

🔄 Training phase starting...
  Epoch 1/1, Batch 1: loss=0.3456

✅ Step 1/25000 completed!
  Overall mean loss: 0.3456
  Mean reward: 0.6250
```

## 📁 Project Structure

### Training Scripts
- `train_fast.py` - **RECOMMENDED** Fast training with LoRA (~1 min/step)
- `train.py` - Original full model training (slower, ~9 min/step)
- `test_generation.py` - Test model format before training

### Evaluation
- `evaluate.py` - Evaluate trained models on GSM8K test set

### Core Components
- `grpo_loss.py` - GRPO loss implementation
- `replay_buffer.py` - Experience replay buffer
- `math_tasks.jsonl` - Training dataset (mathematical reasoning tasks)

### Guides
- `FAST_TRAINING_GUIDE.md` - Complete guide for fast training
- `colab_fast_train.md` - Copy-paste ready Colab cells
- `QUICK_START.md` - Quick start guide
- `README.md` - This file

## 📊 Evaluation

After training, evaluate your model on GSM8K test set:

```bash
# Evaluate on full test set (1,319 samples)
python evaluate.py --model_path output_fast/final_merged

# Quick evaluation on 100 samples
python evaluate.py --model_path output_fast/final_merged --num_samples 100

# Evaluate a checkpoint
python evaluate.py --model_path output_fast/step_50
```

The evaluation script:
- Automatically detects and loads LoRA checkpoints
- Uses the same prompt format as training
- Reports accuracy and sample predictions
- Saves results to JSON

Expected accuracy on GSM8K:
- Base Qwen2.5-0.5B: ~10-15%
- After 100 GRPO steps: ~30-35%
- After 500 GRPO steps: ~35-40%

## 🔧 Configuration

### train_fast.py (Recommended)

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
use_lora = True         # Train with LoRA
use_quant = False       # 4-bit quantization (set True if memory issues)
batch_size = 8          # Increased from train.py
num_rollout = 2         # Reduced from train.py for speed
num_prompts = 500       # Limit dataset size
max_steps = 100         
learning_rate = 1e-5    # Like TRL GRPOTrainer
temperature = 1.0
clip_epsilon = 0.28     # Like TRL GRPOTrainer
kl_weight = 0.0         # Beta = 0
```

### train.py (Original, slower)

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
batch_size = 4
num_rollout = 4
temperature = 0.3
max_steps = 100
```

### Recommended Models

Models that work well with this codebase (tested to follow `<answer>` format):
- ✅ `Qwen/Qwen2.5-0.5B-Instruct` - Fast, good for testing
- ✅ `Qwen/Qwen2.5-1.5B-Instruct` - Better reasoning
- ✅ `HuggingFaceTB/SmolLM2-1.7B-Instruct` - Alternative option

Models that may NOT work (don't follow the format):
- ❌ `LiquidAI/LFM2-350M` - Doesn't generate `<answer>` tags

## 📚 Sources

- https://github.com/mingyin0312/RLFromScratch/blob/main/grpo_train_from_scratch.py
- https://github.com/open-thought/tiny-grpo/blob/main/train.py
- https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py