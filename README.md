# RIS-LLM: Reasoning-Informed Semantic Large Language Model for Electricity Price Forecasting

The implementation of **RIS-LLM**, a reasoning-informed semantic framework that bridges the gap between high-dimensional numerical time-series data and physical market drivers for electricity price forecasting (EPF).

## Overview

Electricity prices are high-level indicators of modern energy system operations, but traditional models struggle to dynamically capture multi-scenario influencing factors under high-volatility market conditions. RIS-LLM addresses this through three key innovations:

1. **Reasoning Feature Extraction (RFE)** — Volatility-aware adaptive LOESS decomposition that disentangles transient fluctuations from structural trends, then extracts morphological features via Important Data Points (IDPs).
2. **Time Series Semantics (TSS) Alignment** — Causal-bias attention guided by pairwise Granger causality, plus unsupervised semantic labeling that assigns interpretable labels (e.g., *Demand Surge*, *Renewable Oversupply*, *Fossil Oversupply*) to price fluctuation patterns.
3. **LLM Reasoning Framework** — Reprogramming a frozen LLaMA-7B via structured prefix prompts and prototype-based cross-attention, enabling the LLM to reason over causally-structured semantic features.

### Architecture

```
Input Price Series [B, L, D]
    │
    ├──► RevIN Normalization
    │
    ├──► RFE Module
    │     ├── Rolling volatility estimation
    │     ├── Adaptive LOESS (Nadaraya-Watson kernel estimator)
    │     ├── IDP detection (local extrema)
    │     └── Morphological features: magnitude, duration, slope, area
    │     Output: M_feat [B, L, 5D]
    │
    ├──► TSS Module
    │     ├── Causal-Bias Attention (λ-learnable, Granger prior C)
    │     └── K-Means semantic labeling at price-fluctuation IDP positions
    │     Output: M_attn [B, L, D'], semantic labels L₁..Lₙ
    │
    ├──► Framework
    │     ├── Overlapping patching (stride < patch_len)
    │     ├── Prototype selection (cosine similarity, top-k)
    │     ├── Cross-attention: patch queries → prototype KV
    │     └── 4-component structured prompt: DataDesc + TaskInst + PatchInfo + PatchLabels
    │
    └──► Frozen LLaMA-7B → Output Projection → De-normalize → Predictions [B, H, D]

All intermediate data in bfloat16 precision.
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (24GB+ VRAM recommended for LLaMA-7B with 4-bit quantization)


## Installation

```bash
pip install -r requirements.txt
```

### LLaMA Model Weights

Place LLaMA-7B HuggingFace-format weights in the `llama/` directory:

```
llama/
├── config.json
├── tokenizer.model
└── pytorch_model-*.bin  (or model.safetensors)
```

## Usage

### Training RIS-LLM

```bash
bash scripts/train.sh
```

### Training Time-LLM (Baseline)

```bash
python main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model TimeLLM \
    --data EPF \
    ...
```

## Project Structure

```
RIS-LLM/
├── main.py                         # Training script
├── config.py                       # Argument definitions
├── risllm/                         # RIS-LLM core implementation
│   ├── model.py                    # RISLLMModel: full pipeline assembly
│   ├── rfe.py                      # RFE: ALOESS + IDP + morphology
│   ├── tss.py                      # TSS: causal attention + semantic labeling
│   ├── granger.py                  # Pairwise Granger causality precompute
│   ├── causal_attention.py         # Causal-Bias Attention (λ-learnable)
│   ├── prototype.py                # PrototypeBank + cross-attention fusion
│   ├── prompt.py                   # 4-component structured prompt builder
│   └── framework.py                # Patch statistics computation
├── models/
│   └── TimeLLM.py                  # Time-LLM baseline
├── layers/
│   ├── Embed.py                    # Token/Patch embedding layers
│   └── StandardNorm.py             # RevIN normalization
├── data_provider/
│   ├── data_loader.py              # Dataset classes (ETT, Custom, M4)
│   ├── data_factory.py             # Data provider factory
│   └── m4.py                       # M4 dataset support
├── utils/
│   ├── tools.py                    # Training utilities, early stopping
│   ├── metrics.py                  # MSE, MAE, RMSE, MAPE
│   ├── timefeatures.py             # Time feature encoding
│   └── losses.py                   # sMAPE, MAPE, MASE losses
├── dataset/                        # Data files
│   ├── spain_epf.csv
│   └── prompt_bank/
├── llama/                          # LLaMA weights placeholder
├── scripts/
│   └── train.sh                    # Example training command
├── checkpoints/                    # Saved models
├── results/                        # Prediction outputs
└── README.md
```

## License

This project is provided for academic and research purposes. See the license file for details.
