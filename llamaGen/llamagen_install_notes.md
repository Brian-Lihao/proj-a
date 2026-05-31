# LlamaGen task environment (Python 3.10, CUDA 12.1)

## 1) Create env
```bash
conda create -n gcpo_llamagen python=3.10 -y
conda activate gcpo_llamagen
python -m pip install -U pip setuptools wheel
```

## 2) Install PyTorch first (official cu121 wheels)
```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## 3) Install main requirements
```bash
python -m pip install -r llamagen_requirements_clean.txt
```

## 4) Optional / evaluation-only extras
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

## 5) Required model assets (not Python packages)
- jadohu/LlamaGen-T2I
- google/flan-t5-xl (must include PyTorch/safetensors weights, not only TF/Flax files)
- vq_ds16_t2i.pt
- HPS_v2.1_compressed.pt (if using HPS reward)
