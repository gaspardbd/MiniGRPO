# Guide d'Entraînement Rapide avec LoRA

## Problème résolu


(`train_fast.py`): **10x plus rapide**
- LoRA : seulement ~1% des paramètres entraînés
- Quantization 4-bit optionnelle
- Pas de reference model séparé (utilise les log probs actuels)
- Optimisé comme votre TP avec GRPOTrainer

## 📊 Comparaison avec votre TP

| Aspect | Votre TP (GRPOTrainer) | train_fast.py |
|--------|----------------------|---------------|
| LoRA | ✅ Oui | ✅ Oui |
| Quantization | ✅ 4-bit | ✅ 4-bit (optionnel) |
| Reference model | Implicite | Pas de modèle séparé |
| Implémentation | HuggingFace TRL | Custom (votre code) |
| Rewards | 2 fonctions séparées | 1 fonction combinée |
| Contrôle | Limité | Total |

## Utilisation rapide

### 1. Dans Colab

```python
%cd /content/MiniGRPO
!git pull origin main

# Installer PEFT si nécessaire
!pip install peft bitsandbytes

# Lancer l'entraînement rapide
!python train_fast.py
```

### 2. Localement

```bash
cd MiniGRPO
pip install peft bitsandbytes
python train_fast.py
```

## ⚙️ Configuration par défaut

```python
# Modèle
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
use_lora = True
use_quant = False  # Mettez True si problème de mémoire

# Dataset
num_prompts = 500  # Au lieu de 100k
batch_size = 8     # Augmenté de 4 à 8
num_rollout = 2    # Réduit de 4 à 2

# Training
max_steps = 100
learning_rate = 1e-5
clip_epsilon = 0.28  # Comme votre TP
kl_weight = 0.0      # Beta = 0 comme votre TP

# LoRA
lora_r = 64
lora_alpha = 128
lora_dropout = 0.1
```

## 📈 Vitesse attendue

| Configuration | Temps par step | 100 steps |
|--------------|----------------|-----------|
| train.py (original) | ~9 min | ~15h |
| train_fast.py (LoRA) | ~1 min | ~1h40 |
| train_fast.py (LoRA + quant) | ~45 sec | ~1h15 |

## 💾 Checkpoints et évaluation

### Sauvegardes

Le script sauvegarde :
- `output_fast/step_25/` - Checkpoint à 25 steps
- `output_fast/step_50/` - Checkpoint à 50 steps
- `output_fast/step_75/` - Checkpoint à 75 steps
- `output_fast/final/` - LoRA adapters finaux
- `output_fast/final_merged/` - Modèle mergé (LoRA + base)

### Évaluation

```bash
# Évaluer sur tout le test set GSM8K
python evaluate.py --model_path output_fast/final_merged

# Évaluer sur 100 samples seulement
python evaluate.py --model_path output_fast/final_merged --num_samples 100

# Évaluer un checkpoint intermédiaire
python evaluate.py --model_path output_fast/step_50
```

Le script d'évaluation :
- ✅ Détecte automatiquement si c'est un checkpoint LoRA
- ✅ Merge les adapters automatiquement
- ✅ Utilise le même format de prompt que l'entraînement
- ✅ Calcule l'accuracy sur GSM8K
- ✅ Sauvegarde les résultats en JSON

### Exemple de sortie d'évaluation

```
============================================================
EVALUATION RESULTS
============================================================
Accuracy: 34.52%
Correct: 453/1319
============================================================

📝 Sample predictions:

Example 1:
  Question: Janet's ducks lay 16 eggs per day...
  True answer: 18
  Generated: 18
  Correct: ✓
```

## 🔧 Personnalisation

### Pour un training plus long

Modifiez dans `train_fast.py` :

```python
num_prompts = 2000   # Plus de données
max_steps = 500      # Plus de steps
```

### Pour plus de vitesse

```python
use_quant = True     # Active la quantization 4-bit
num_rollout = 1      # Réduit à 1 rollout
batch_size = 16      # Augmente le batch
max_length = 200     # Réduit la longueur max
```

### Pour éviter OOM (Out of Memory)

```python
use_quant = True           # Active la quantization
batch_size = 4             # Réduit le batch
num_rollout = 1            # Réduit les rollouts
max_length = 150           # Réduit la longueur
gradient_accumulation = 2  # Ajouter accumulation
```

## 🐛 Troubleshooting

### Erreur: "CUDA out of memory"

```python
# Dans train_fast.py, modifiez:
use_quant = True
batch_size = 2
num_rollout = 1
```

### Erreur: "peft not found"

```bash
pip install peft bitsandbytes
```

### Rewards toujours à 0

Le modèle ne génère pas le bon format. Lancez :
```bash
python test_generation.py
```

### Loss explose (>100)

```python
# Réduisez le learning rate
learning_rate = 5e-6  # Au lieu de 1e-5
```

## 📊 Différences avec train.py original

| Feature | train.py | train_fast.py |
|---------|----------|---------------|
| LoRA | ❌ Non | ✅ Oui |
| Reference model | ✅ Modèle séparé sur CPU | ❌ Utilise log probs actuels |
| Vitesse | ❌ 9 min/step | ✅ 1 min/step |
| Mémoire | ❌ Haute | ✅ Basse |
| Dataset | 100k prompts | 500 prompts (configurable) |
| Batch size | 4 | 8 |
| Rollouts | 4 | 2 |

## 🎯 Workflow complet

```bash
# 1. Test rapide du modèle
python test_generation.py

# 2. Entraînement rapide
python train_fast.py

# 3. Évaluation
python evaluate.py --model_path output_fast/final_merged --num_samples 100

# 4. Si bon, évaluation complète
python evaluate.py --model_path output_fast/final_merged
```

## 📝 Notes

- **LoRA** : Entraîne seulement des matrices de rang faible (~1% des paramètres)
- **No ref model** : On utilise les log probs de la politique actuelle comme référence au lieu d'un modèle séparé (simplifie et accélère)
- **Quantization** : Optionnelle, réduit la mémoire de 4x
- **num_prompts** : Limité à 500 par défaut pour la vitesse, augmentez selon vos besoins

Bon entraînement rapide ! 🚀

