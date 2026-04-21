# Aggregated Window Attention (AWA)

A general sparse attention architecture for scalable, interpretable long-range sequence modeling. Built on the [Swin Transformer](https://arxiv.org/pdf/2103.14030), AWA replaces the standard MLP block with a fused global attention mechanism that achieves approximately linear complexity in sequence length.

We explore AWA in computer vision as a testbed, but the architecture is domain-agnostic and designed for any task where long-range dependencies matter and dense attention is too expensive.

---

## How It Works

Each **Aggregated Window Attention Block (AWAB)** modifies the standard transformer block in two ways:

1. **Local attention** — shifted windowed attention (from Swin Transformer), with an extra token **T** prepended to each window.
2. **Fused global attention via AWA** — after local attention, the MLP step is augmented to enable global communication between windows.

### AWA-Fused MLP (step by step)

| Step | Operation |
|------|-----------|
| 1 | Shifted windowed attention + residual. Each window has an extra token **T**. |
| 2 | Expand each token from dim `d` → `4d` (first linear of MLP). |
| 3 | Pass **T** through a linear layer + softmax → **AWV** (Aggregation Weight Vector, length `m`). |
| 4 | Weighted sum of expanded tokens in the window using AWV → **AWT** (Aggregated Window Token, dim `4d`). |
| 5 | Dense attention between all AWTs across windows. |
| 6 | Concatenate **T** with the AWT value token, pass through linear + softmax → **DWV** (Disaggregation Weight Vector, length `m`). |
| 7 | Broadcast the AWT value back to all tokens in the window, scaled by DWV. |
| 8 | Contract from `4d` → `d` (second linear of MLP) + residual. |

Many of the compute-heavy steps (token expansion/contraction, attention-based aggregation) are shared with the standard MLP, so the overhead of global communication is small.

### Complexity

```
FLOPs_AWAB  = 24nd² + 4mnd + 96wd² + 24nd + 16w²d
FLOPs_dense = 24nd² + 4n²d
```

where `n` = tokens, `d` = embedding dim, `w` = number of windows, `m` = tokens per window. AWA scales **~linearly** in `n`; dense attention scales quadratically.

---

## Architecture

The model ([src/model/v1/model.py](src/model/v1/model.py)) is a hierarchical 4-stage vision transformer:

- **Patch embedding** — 4×4 non-overlapping patches
- **4 stages** of `SwinTransformerBlock` layers, each using the AWA-fused MLP
- **Patch merging** between stages (halves spatial resolution, doubles channels)
- **Classification head** — global average pool → linear

The AWA logic lives entirely in the `Mlp` class: AWV generation (`get_AWV`), dense attention between AWTs (`dense_attn`, a `WindowAttention` module treating all windows as a single sequence), and DWV generation (`get_DWV`).

---

## Project Structure

```
src/
  model/v1/
    model.py          # SwinTransformer, Mlp (AWA-fused), WindowAttention, PatchEmbed
  train/
    train.py          # Step-based training loop with gradient accumulation
  config/
    config.py         # DataConfig, HyperParamConfig, ModelConfig, RunConfig
  data_io/
    data_loader.py    # Dataset + DataLoader construction (Tiny ImageNet)
    download.py       # Dataset download/setup
  utils/
    train_helpers.py  # LR schedule (cosine with warmup)
    checkpoint.py     # Save/load checkpoints
  log/
    setup.py          # Logger setup
```

---

## Training

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
python -m src.train.train
```

Training is configured via [src/config/config.py](src/config/config.py). Key settings:

| Config | Field | Notes |
|--------|-------|-------|
| `DataConfig` | `gpu_batch_size` | Per-GPU batch size |
| `HyperParamConfig` | `total_batch_size` | Effective batch size (enables gradient accumulation) |
| `HyperParamConfig` | `total_train_steps` | Total optimizer steps |
| `HyperParamConfig` | `constant_lr` | Set to `None` to use cosine schedule |
| `HyperParamConfig` | `max_lr` | Peak LR for cosine schedule |
| `HyperParamConfig` | `min_lr` | Floor LR for cosine schedule |
| `ModelConfig` | `embed_dim` | Base channel width |
| `ModelConfig` | `depths` | Transformer blocks per stage |
| `RunConfig` | `val_interval` | Steps between validation runs |

The LR schedule is cosine decay with linear warmup over `warmup_fraction` of total steps.

---

## Results

Preliminary results on Tiny ImageNet (200 classes) after 6,250 training steps:

| Model | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| baseline (AWA) | 0.4907 | 0.4092 | 2.9891 |
| no_awa (vanilla MLP) | 0.4404 | 0.3755 | 3.1478 |

AWA improves val accuracy by **+3.4 pp** over the vanilla Swin MLP baseline in the same number of steps, confirming that the global attention mechanism provides a meaningful learning signal beyond local windowed attention alone.

---

## Key Design Insight

Most local-attention architectures that add global tokens keep them at the same dimensionality `d` as local tokens, making them information bottlenecks. AWA expands global tokens to `4d` via the shared MLP expansion before dense attention — relying on **superposition** to pack richer inter-region information into each AWT without extra parameters for the expansion.

This also improves interpretability: dense attention over a small number of window-level tokens is easier to analyze than attention over all individual tokens. Pairs of windows with large cross-attention can then be examined further via the local attention maps, AWVs, and DWVs.


## Acknowledgements

Local windowed attention and the hierarchical backbone are adapted from the [Swin Transformer](https://github.com/microsoft/Swin-Transformer) (Liu et al., ICCV 2021).
