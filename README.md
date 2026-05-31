# Balancing Performance and Diversity in GRPO Autoregressive Text-to-Image Post-Training

![](./assets/image-1.png)

## Janus task environment

### 1) Create env
```bash
conda create -n gcpo_janus python=3.10 -y
conda activate gcpo_janus
pip install --upgrade pip setuptools wheel
```

### 2) Install PyTorch first
For CUDA 12.1:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 3) Install main requirements
```bash
pip install -r janus_requirements.txt
```

### 4) Install FlashAttention separately
```bash
pip install flash_attn==2.6.1 --no-build-isolation
```

### 5) ImageReward

Recommended separate env:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/zai-org/ImageReward.git
cd ImageReward
pip install -e .
```

## LlamaGen task environment (Python 3.10, CUDA 12.1)

## 1) Create env
```bash
conda create -n gcpo_llamagen python=3.10 -y
conda activate gcpo_llamagen
python -m pip install -U pip setuptools wheel
```

### 2) Install PyTorch first (official cu121 wheels)
```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3) Install main requirements
```bash
python -m pip install -r llamagen_requirements_clean.txt
```

### 4) Optional / evaluation-only extras
- ImageReward package:
```bash
python -m pip install image-reward
```

- If you also want the ImageReward repo scripts or benchmark data:
```bash
git clone https://github.com/THUDM/ImageReward.git
cd ImageReward
python -m pip install image-reward
cd ..
```

### 5) Required model assets (not Python packages)
- jadohu/LlamaGen-T2I
- google/flan-t5-xl (must include PyTorch/safetensors weights, not only TF/Flax files)
- vq_ds16_t2i.pt
- HPS_v2.1_compressed.pt (if using HPS reward)

## 🚀 Train

### LlamaGen

```bash
cd llamaGen/src
bash scripts/rl_gcpo_hps.sh
```

### Janus-Pro

```bash
cd janus/src
bash scripts/run_gcpo_hps.sh
bash scripts/run_gcpo_geneval.sh
```

## 💫 Inference

### LlamaGen

```bash
cd llamaGen
bash scripts/inference.sh
```

### Janus-Pro

```bash
cd janus/src
python gcpo/src/infer/infer.py
```

