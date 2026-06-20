# ICON-NET: Integrated Channel-Spatial and Gate Attention Network for Brain Tumor Segmentation

ICON-NET is a dual-attention encoder-decoder architecture for MRI brain tumor segmentation. It combines **CBAM** (channel-spatial attention) inside every ConvBlock of the encoder and decoder with **AttentionGate**-based filtering at every decoder skip connection, coordinated with deep supervision and a hybrid Dice + BCE + Focal loss.

> Conv or Volve? Why Not Both! Parallel Networks for Brain Tumor Segmentation
> Ahmad Nafees Tihami, Md. Ashikul Islam, Sowad Rahman — BRAC University

## Key Results

Evaluated with a 5-seed × 5-fold stratified cross-validation protocol (25 independent runs per dataset, 75 total) on three public MRI datasets.

| Dataset | Modality / Grade | Best Dice (%) | Mean Dice ± Std (%) | Best Precision (%) |
|---|---|---|---|---|
| BRISC 2025 | T1-weighted, HGG | 87.49 | 86.16 ± 1.82 | 88.45 |
| BraTS 2021 | T1ce, HGG | 83.70 | 82.77 ± 0.82 | 85.40 |
| LGG Segmentation | FLAIR, LGG | 84.27 | 81.14 ± 2.89 | 86.01 |

ICON-NET is trained entirely from scratch (no pretrained backbone), with 33.22M parameters and 121.53 GFLOPs, and is benchmarked against 10 CNN-, Transformer-, Mamba-, and foundation-model-based baselines (UNet, TransUNet, SwinUNet, UCTransNet, MambaUNet, VMUNet, SegMamba, H2Former, CATNet, MedSAM).

## Architecture

- 4-level symmetric encoder-decoder (UNet-style), base channel width 64 (64→128→256→512), 1024-channel bottleneck at 16×16.
- **ConvBlock**: Conv3×3 → BN → ReLU → Dropout2d(0.3) → Conv3×3 → BN → ReLU → Dropout2d(0.3) → CBAM → residual addition.
- **CBAM**: channel attention (shared MLP, reduction ratio r=16, GAP+GMP) followed by spatial attention (7×7 conv over channel-pooled maps).
- **AttentionGate**: gating signal from the deeper decoder stage filters each encoder skip connection before concatenation (intermediate dimension Fi = Cout/2).
- **Deep supervision**: auxiliary 1×1 conv heads at decoder levels d4, d3, d2, weighted 0.6 (main) + 0.2 + 0.1 + 0.1.
- **Loss**: 0.5·Dice + 0.3·BCE + 0.2·Focal (γ=2, α=0.25).
- **Training**: AdamW (lr=1e-4, wd=1e-5), CosineAnnealingWarmRestarts (T0=10, T_mult=2), batch size 16, max 80 epochs, early stopping (patience 7, val Dice), input 256×256×3, NVIDIA Tesla T4 (15GB).

See the paper PDF for full derivations, ablations, and qualitative results.

## Repository Structure

```
ICON-NET/
├── model/
│   └── Icon_net.ipynb              # Core ICON-NET architecture definition
├── experiments/                    # Main 5-seed x 5-fold cross-validation runs
│   ├── BRISC/                      # BRISC 2025 — 25 runs (seed42/123/456/789/2024 x fold1-5)
│   ├── BraTS/                      # BraTS 2021 — 25 runs
│   └── LGG/                        # LGG Segmentation — 25 runs
├── baselines/                      # Single-run SOTA comparison models
│   ├── BRISC/                      # UNet, TransUNet, SwinUNet, UCTransNet, MambaUNet,
│   ├── BraTS/                      # VMUNet, SegMamba, H2Former, CATNet, MedSAM
│   └── LGG/
└── ablation/                       # Component-wise ablation studies
    ├── arch/                       # CBAM / AttentionGate / DeepSupervision / Residual removal, placement
    ├── AttentionGate/               # Fi dimension variants (Cout, Cout/4)
    ├── batch/                      # Batch size 4, 8
    ├── deep_supervision/            # Alternative DS weight schemes
    ├── dropout/                    # Dropout rate variants (LGG)
    ├── loss/                       # Single-loss and reweighted loss variants
    ├── lr/                         # Learning rate 1e-2, 1e-3, 1e-4
    └── optimizer/                  # SGD, Lion vs AdamW
```

## Datasets

| Dataset | Tumor Type | Modality | Samples | Source |
|---|---|---|---|---|
| BRISC 2025 | HGG | T1-weighted | 4,793 slices | [Fateh et al.](https://arxiv.org/abs/2506.14318) |
| BraTS 2021 | HGG | T1ce | 5,000-slice subset | [Official release](https://arxiv.org/abs/2107.02314) · [2D slices used here](https://www.kaggle.com/datasets/ahmadnafeestihami/brats2021-2d-slices) |
| LGG Segmentation | LGG | FLAIR | 3,929 slices | TCIA · [2D slices used here](https://www.kaggle.com/datasets/ahmadnafeestihami/lgg2d-slices) |

All datasets are resized to 256×256 and stored as 3-channel images.

## Ablation Highlights (BRISC 2025, seed=2024, fold=2, baseline Dice = 87.19%)

| Component removed | ΔDice |
|---|---|
| Residual connection | −7.94% |
| CBAM | −3.23% |
| CBAM + AttentionGate | −3.95% |
| Deep Supervision | −0.90% |
| AttentionGate | −0.61% |

Full ablation tables (loss weighting, CBAM placement, gating source, deep-supervision weights, dropout, optimizer, AttentionGate bottleneck size, batch size, learning rate, network depth) are in the paper, Section IV-G.

