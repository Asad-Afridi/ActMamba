# ActMamba

## Overview
> The source code, benchmark datasets, pretrained models, and detailed documentation will
> be publicly released upon completion of the editorial/review process of
> the associated manuscript.


**ActMamba: Adaptive State-Space Parameter Modulation for Conditional Information Fusion in Temporal Action Segmentation**

ActMamba is a linear-time conditional diffusion framework for temporal action segmentation (TAS) that reformulates conditional information fusion through Adaptive State-Space Parameter Modulation (ASPM). The framework introduces:

- A novel Gated Cross Contextual (GC²)-Mamba module for adaptive state-space parameter modulation
- A hierarchical multi-scale Mamba encoder–decoder for long-range and fine-grained temporal modeling
- A learnable mask embedding for masked contextual features during diffusion training
- DDIM-based denoising for action label generation

Compared with DiffAct++, ActMamba reduces FLOPs by 47.9%, accelerates inference by 13.4×, and lowers GPU memory consumption by 8.4×. These gains stem from replacing quadratic cross-attention with linear-time state-space parameter modulation.

---


## Features

- Bidirectional Mamba encoder with multi-scale dilated convolutions
- GC²-Mamba decoder with Adaptive State-Space Parameter Modulation (ASPM)
- DDIM-based conditional diffusion for iterative action label refinement
- Linear-time complexity (vs. quadratic for attention-based methods)
- Extensive evaluation on 50Salads, GTEA, and Breakfast benchmarks

---

## Architecture

- **Encoder**: Bidirectional Mamba with multi-scale dilated convolutions for hierarchical temporal feature extraction
- **Decoder**: GC²-Mamba — generates selective state-space parameters (Δ, B, C) from the evolving decoder representation and adaptively modulates them using contextual encoder features via learned gating
- **Diffusion**: DDIM-based denoising for action label generation (inherited from DiffAct)

---

## Dataset

ActMamba is evaluated on three standard temporal action segmentation benchmarks:

- [GTEA](https://ieeexplore.ieee.org/abstract/document/5995444) — Georgia Tech Egocentric Activities
- [50Salads](https://dl.acm.org/doi/abs/10.1145/2493432.2493482) — 50 Salads activity recognition
- [Breakfast](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Kuehne_The_Language_of_2014_CVPR_paper.html) — Breakfast Actions Dataset

Download the I3D features and ground truth labels from the links provided in [MS-TCN](https://github.com/yabufarha/ms-tcn) or [ASFormer](https://github.com/ChinaYi/ASFormer).

Organize the data under a `datasets/` directory:

```
datasets/
├── gtea/
│   ├── features/          # .npy files (I3D features)
│   ├── groundTruth/       # .txt files (frame-level labels)
│   ├── mapping.txt        # action index to name mapping
│   └── splits/            # train/test split bundles
│       ├── train.split1.bundle
│       └── test.split1.bundle
├── 50salads/
│   └── ...
└── breakfast/
    └── ...
```

## Installation

```bash
git clone https://github.com/Asad-Afridi/ActMamba.git
cd ActMamba

pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch >= 2.0
- [mamba-ssm](https://github.com/state-spaces/mamba) (`pip install mamba-ssm`)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) >= 1.1.0 (`pip install causal-conv1d>=1.1.0`)

```bash
pip install torch numpy scipy tqdm matplotlib tensorboard einops
pip install mamba-ssm causal-conv1d>=1.1.0
```

---

## Training

Train a single split:

```bash
python main.py --config configs/gtea.json --split 1 --device 0
python main.py --config configs/50salads.json --split 3 --device 0
python main.py --config configs/breakfast.json --split 1 --device 0
```

Train all splits sequentially:

```bash
python scripts/train_all_splits.py --config configs/gtea.json --device 0
```

Training progress is logged via TensorBoard under the `result/` directory. Best models are automatically saved to `chkpnt/{dataset_name}/split{N}.model`.

---

## Evaluation

Evaluate a single split using a saved checkpoint:

```bash
python main.py --config configs/gtea.json --split 1 --test --device 0
```

Evaluate all splits and compute averaged metrics:

```bash
python main.py --config configs/gtea.json --split 0 --test --device 0
```

Results are printed to stdout and saved to `result/{dataset_name}/`.

---

## Project Structure

```
├── main.py                     # Training and evaluation entry point
├── configs/                    # Dataset-specific configuration files
│   ├── gtea.json
│   ├── 50salads.json
│   └── breakfast.json
├── core/
│   ├── __init__.py
│   ├── dataset.py              # Data loading and augmentation
│   ├── trainer.py              # Training loop and evaluation
│   ├── utils.py                # Metrics (Edit, F1) and utilities
│   ├── visualization.py        # Prediction visualization
│   └── model/
│       ├── __init__.py
│       ├── actmamba.py         # Main model with diffusion
│       ├── backbone.py         # Mamba encoder and decoder modules
│       ├── layers.py           # Multi-scale Mamba layers
│       └── cross_mamba.py      # GC²-Mamba (cross-contextual SSM)
└── scripts/
    ├── train_all_splits.py     # Train all splits sequentially
    ├── compare_runs.py         # Compare results across runs
    ├── generate_configs.py     # Regenerate config files
    └── profile_model.py        # Speed and memory profiling
```

---

## Model Profiling

Profile inference speed, memory usage, and GFLOPs:

```bash
pip install thop  # optional, for FLOP counting
python scripts/profile_model.py --config configs/gtea.json --device 0
```

---

## Citation

If you use this repository, please cite our paper.

```bibtex
Citation will be added after publication.
```

---

## Acknowledgements

This codebase is built upon [DiffAct](https://github.com/Finspire13/DiffAct) by Daochang Liu et al. We thank the authors for making their code publicly available.

> Daochang Liu, Qiyue Li, AnhDung Dinh, Tingting Jiang, Mubarak Shah, Chang Xu. "Diffusion Action Segmentation." *ICCV 2023*.

The Mamba components are based on the [mamba-ssm](https://github.com/state-spaces/mamba) library by Albert Gu and Tri Dao.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---


## Contact

**Asad Ullah**

Department of Applied Artificial Intelligence

Sungkyunkwan University

Email: *asadullah@g.skku.edu*

---
