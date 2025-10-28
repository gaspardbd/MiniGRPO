# 🚀 Colab Fast Training - Copy-Paste Ready

Copiez ces cellules dans votre notebook Colab dans l'ordre.

## Cellule 1: GPU Check

```python
!nvidia-smi
```

## Cellule 2: Clone & Update

```bash
%%bash
cd /content
if [ ! -d "MiniGRPO" ]; then
  git clone https://github.com/gaspardbd/MiniGRPO.git
fi
cd MiniGRPO
git pull origin main
ls -la
```

## Cellule 3: Install Dependencies

```bash
%%bash
pip install -q torch transformers accelerate datasets
pip install -q peft bitsandbytes  # For LoRA and quantization
pip install -q tqdm
```

## Cellule 4: HuggingFace Login

```python
from huggingface_hub import login
login("hf_xxxxx")  # Replace with your token
```

## Cellule 5: Test Model First (Optional but Recommended)

```python
%cd /content/MiniGRPO
!python test_generation.py
```

Vérifiez que vous voyez des `✅ Found <answer> tag!`

## Cellule 6: Launch Fast Training 🚀

```python
%cd /content/MiniGRPO
import os
os.environ['PYTHONUNBUFFERED'] = '1'

# Run the fast training script
%run train_fast.py
```

## Cellule 7: Monitor Progress (Alternative - in separate notebook cell)

Si vous voulez monitorer pendant que ça tourne :

```python
# Dans une nouvelle cellule pendant que training.py tourne
import time
while True:
    !tail -20 /content/MiniGRPO/training.log 2>/dev/null || echo "No log yet"
    time.sleep(30)  # Update every 30 seconds
```

## Cellule 8: Evaluate After Training

```python
%cd /content/MiniGRPO

# Quick eval on 100 samples
!python evaluate.py --model_path output_fast/final_merged --num_samples 100

# Full evaluation (takes longer)
# !python evaluate.py --model_path output_fast/final_merged
```

## Cellule 9: Download Checkpoint

```bash
%%bash
cd /content/MiniGRPO
zip -r final_model.zip output_fast/final_merged/
```

```python
from google.colab import files
files.download('/content/MiniGRPO/final_model.zip')
```

## 🎯 What to Expect

### Training Logs

```
============================================================
FAST TRAINING CONFIGURATION
============================================================
Model: Qwen/Qwen2.5-0.5B-Instruct
LoRA: True, Quantization: False
Loaded 500 prompts (limited from full dataset)
Batch size: 8
Rollouts per sample: 2
Steps to run: 100
============================================================

============================================================
Step 1/100 (1.0%): starting rollouts
============================================================
  Sample 1/8: generating 2 rollouts...
  Sample 1: done. Rewards: [0.5, 1.0]
  Sample 2/8: generating 2 rollouts...
  Sample 2: done. Rewards: [1.0, 1.0]
  ...

📊 Rollout Summary:
  Buffer size: 16
  Mean reward: 0.6875
  Min/Max: 0.00/1.00

🔄 Training phase...
  Epoch 1, Batch 1: loss=0.4523
  Epoch 1, Batch 2: loss=0.3891

✅ Step 1/100 completed!
  Mean loss: 0.4207
  Mean reward: 0.6875
```

### Evaluation Results

```
============================================================
EVALUATION RESULTS
============================================================
Accuracy: 32.45%
Correct: 32/100
============================================================
```

## ⏱️ Time Estimates

- **100 steps with LoRA**: ~1h 40min
- **100 steps with LoRA + quant**: ~1h 15min
- **Evaluation (100 samples)**: ~5 minutes
- **Evaluation (full test set)**: ~30-40 minutes

## 🔧 Quick Modifications

### For even faster testing (trade quality for speed)

Before running cell 6, modify `train_fast.py`:

```python
# In train_fast.py, change:
num_prompts = 100      # Instead of 500
max_steps = 20         # Instead of 100
batch_size = 16        # Instead of 8
num_rollout = 1        # Instead of 2
```

### For better quality (trade speed for quality)

```python
num_prompts = 2000     # More data
max_steps = 500        # More training
temperature = 0.8      # Less random
```

### If you get OOM (Out of Memory)

```python
use_quant = True       # Enable 4-bit quantization
batch_size = 4         # Reduce batch size
num_rollout = 1        # Reduce rollouts
```

## 🆚 Comparison: Original vs Fast

| Metric | train.py | train_fast.py |
|--------|----------|---------------|
| Time per step | ~9 min | ~1 min ⚡ |
| Parameters trained | 500M (100%) | ~5M (1%) 💾 |
| Memory usage | High | Low 📉 |
| 100 steps time | ~15 hours | ~1.7 hours 🚀 |

## ✅ Success Indicators

- ✅ Rewards > 0 (ideally 0.4-0.8)
- ✅ Loss decreasing (starting ~1-3, ending ~0.2-0.8)
- ✅ Mean reward increasing over steps
- ✅ Evaluation accuracy > 25% (random is ~0%)

## 📊 Expected Accuracy Progress

- **Before training** (base Qwen2.5-0.5B): ~10-15%
- **After 25 steps**: ~20-25%
- **After 50 steps**: ~25-30%
- **After 100 steps**: ~30-35%
- **After 500 steps**: ~35-40%

(Note: Ces chiffres sont des estimations, peuvent varier)

---

🎉 **Vous êtes prêt !** Copiez les cellules dans Colab et lancez !

