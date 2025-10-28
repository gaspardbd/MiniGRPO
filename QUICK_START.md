# 🚀 Guide de Démarrage Rapide - Google Colab

## ✅ Modifications effectuées

Le problème a été identifié et corrigé :
- ❌ **Avant** : `LiquidAI/LFM2-350M` ne générait pas les balises `<answer>` → rewards à 0
- ✅ **Maintenant** : `Qwen/Qwen2.5-0.5B-Instruct` suit correctement le format

## 📋 Instructions pour relancer dans Colab

### 1. **Stoppez l'entraînement actuel**
Dans votre notebook Colab, cliquez sur le bouton STOP (⏹️) ou utilisez Runtime → Interrupt execution

### 2. **Mettez à jour le code**
Exécutez cette cellule dans Colab :

```python
%cd /content/MiniGRPO
!git pull origin main
```

### 3. **[OPTIONNEL] Testez le nouveau modèle d'abord**
Pour vérifier que Qwen génère bien les balises `<answer>` :

```python
%cd /content/MiniGRPO
!python test_generation.py
```

Vous devriez voir des sorties comme :
```
✅ Found <answer> tag!
   Predicted: '5'
   Expected: '5'
   Match: True
```

### 4. **Relancez l'entraînement**
Choisissez une des deux méthodes :

**Option A : Python (Recommandé)**
```python
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.chdir('/content/MiniGRPO')
%run train.py
```

**Option B : Bash**
```bash
%%bash
cd /content/MiniGRPO
export PYTHONUNBUFFERED=1
python -u train.py 2>&1 | tee train.log
```

## 📊 Ce que vous devriez voir maintenant

### ✅ Configuration au démarrage
```
============================================================
TRAINING CONFIGURATION
============================================================
Model: Qwen/Qwen2.5-0.5B-Instruct
Loaded 100000 prompts
Batch size: 4
Rollouts per sample: 4
Total possible steps: 25000
Steps to run (max_steps): 100
Checkpoint interval: every 20 steps
============================================================
```

### ✅ Pendant l'entraînement
```
============================================================
Step 1/100 (0.0%): starting rollouts
============================================================
  Sample 1/4: generating 4 rollouts...
  
  🔍 DEBUG - First completion sample:
  Question: What is 2 + 3?...
  Expected answer: 5
  Generated completion: <think>To add 2 and 3...</think><answer>5</answer>
  ---
  
  Sample 1: generation done. Rewards: [1.0, 1.0, 0.5, 1.0]  ← Rewards > 0 !

📊 Rollout Summary:
  Buffer size: 16
  Mean reward: 0.8750  ← Bon signe !
  Min/Max reward: 0.50/1.00

🔄 Training phase starting...
  Epoch 1/1, Batch 1: loss=0.3456  ← Loss normale (< 10)
```

## 🎯 Signes de succès

✅ **Rewards > 0** (idéalement 0.3-0.8 au début)
✅ **Loss < 10** (typiquement 0.1-5.0)
✅ **Vitesse** : ~2-5 minutes par step (beaucoup plus rapide qu'avant)
✅ **Debug message** montre des balises `<answer>` dans les générations

## ⚙️ Paramètres actuels

- **max_steps = 100** : L'entraînement s'arrêtera après 100 steps (~3-8 heures)
- Pour un entraînement complet, modifiez dans `train.py` ligne 158 :
  ```python
  max_steps = 5000  # ou None pour aller jusqu'au bout
  ```

## 🐛 Si ça ne marche toujours pas

### Problème : Rewards toujours à 0
**Solution** : Vérifiez le message DEBUG. Si pas de balise `<answer>`, le modèle n'est pas chargé correctement.

### Problème : Loss explosive (> 100)
**Solution** : 
1. Réduisez le learning rate dans `train.py` ligne 173 :
   ```python
   optimizer=torch.optim.AdamW(model.parameters(), lr=5e-5)  # au lieu de 1e-4
   ```

### Problème : Out of memory
**Solution** :
1. Réduisez `batch_size` de 4 à 2 (ligne 148)
2. Réduisez `num_rollout` de 4 à 2 (ligne 154)
3. Réduisez `max_length` de 512 à 256 (ligne 149)

## 📈 Progression attendue

Avec Qwen2.5-0.5B-Instruct, vous devriez voir :
- **Steps 1-10** : Mean reward ~0.2-0.4, loss ~2-5
- **Steps 10-30** : Mean reward ~0.4-0.6, loss ~1-3
- **Steps 30-100** : Mean reward ~0.6-0.8, loss ~0.5-2

Le modèle apprend à mieux formater ses réponses et à résoudre les problèmes mathématiques !

## 💾 Checkpoints

Les modèles sont sauvegardés dans `/content/MiniGRPO/output/` tous les 20 steps :
- `output/step_20/`
- `output/step_40/`
- `output/step_60/`
- etc.

Pour télécharger un checkpoint depuis Colab :
```python
!zip -r checkpoint_step_40.zip output/step_40/
from google.colab import files
files.download('checkpoint_step_40.zip')
```

Bon entraînement ! 🎉

