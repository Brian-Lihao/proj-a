# Janus task environment

## 1) Create env
```bash
conda create -n gcpo_janus python=3.10 -y
conda activate gcpo_janus
pip install --upgrade pip setuptools wheel
```

## 2) Install PyTorch first
For CUDA 12.1:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

## 3) Install main requirements
```bash
pip install -r janus_requirements.txt
```

## 4) Install FlashAttention separately
```bash
pip install flash_attn==2.6.1 --no-build-isolation
```

## 5) ImageReward

Recommended separate env:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/zai-org/ImageReward.git
cd ImageReward
pip install -e .
```

