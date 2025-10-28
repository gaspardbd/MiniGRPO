# MiniGRPO

Re-implementation of GRPO (Group Relative Policy Optimization) from DeepSeekMath and DeepSeekR1.

## 🚀 Quick Start

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

- `train.py` - Main training script with enhanced logging
- `grpo_loss.py` - GRPO loss implementation
- `replay_buffer.py` - Experience replay buffer
- `math_tasks.jsonl` - Training dataset (mathematical reasoning tasks)
- `colab_train.ipynb` - Google Colab notebook for easy training

## 🔧 Configuration

Key parameters in `train.py`:

```python
model_name = "LiquidAI/LFM2-350M"
batch_size = 4
num_rollout = 4
temperature = 0.3
checkpoint_interval = 20
```

## 📚 Sources

- https://github.com/mingyin0312/RLFromScratch/blob/main/grpo_train_from_scratch.py
- https://github.com/open-thought/tiny-grpo/blob/main/train.py
- https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py