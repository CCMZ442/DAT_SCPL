# DAT-SCPL

Official implementation of **DAT-SCPL: Semi-Supervised Small-Defect Detection with Distribution-Aware Thresholding and Confidence Screening**.

This repository accompanies our manuscript submission to *The Visual Computer* and provides the core code, split files, and figure-generation scripts used in our DeepPCB experiments.

---

## Overview

DAT-SCPL is a lightweight pseudo-label quality control strategy for semi-supervised object detection (SSOD) under tiny and class-imbalanced industrial defect scenarios.

The method consists of two key components:

- **DAT (Distribution-Aware Thresholding)**: replaces a fixed global confidence threshold with class-wise quantile-based admission thresholds estimated from teacher score distributions on unlabeled data.
- **SCPL (Small-defect Confidence-based Pseudo-label Screening)**: performs conservative post-fusion screening on pseudo labels using class-wise confidence statistics.

The full framework follows an **offline two-stage self-training pipeline**:

1. Train a teacher detector on labeled data.
2. Freeze the teacher and generate pseudo labels on unlabeled data.
3. Apply DAT, fusion, and SCPL to obtain filtered pseudo labels.
4. Train the student detector on labeled + pseudo-labeled data.

---

## Repository Structure

```text
DAT_SCPL/
├─ code/                     # utility scripts for result checking and experiment running
├─ paper_fig_assets/         # scripts and assets for reproducing paper figures/tables
├─ qual_examples/            # qualitative example metadata
├─ splits/                   # labeled/unlabeled split files for different ratios and seeds
├─ run_dat_scpl.py           # DAT-SCPL main pipeline
├─ run_dat_only.py           # DAT-only baseline
├─ run_fixedtau.py           # fixed-threshold SSOD baseline
├─ requirements.txt          # Python dependencies
├─ LICENSE
└─ README.md
````

---

## Environment

We recommend Python 3.9+ with PyTorch and Ultralytics installed.

Install dependencies with:

```bash
pip install -r requirements.txt
```

If needed, please additionally install versions of `torch`, `torchvision`, and `ultralytics` that match your local CUDA environment.

---

## Dataset Preparation

Experiments are conducted on **DeepPCB**.

Please organize the dataset in a structure compatible with the scripts, for example:

```text
deeppcb_yolo/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ labels/
├─ splits/
│  ├─ r01_seed0/
│  ├─ r01_seed1/
│  ├─ r01_seed2/
│  ├─ r05_seed0/
│  ├─ r05_seed1/
│  ├─ r05_seed2/
│  ├─ r10_seed0/
│  ├─ r10_seed1/
│  ├─ r10_seed2/
│  ├─ r20_seed0/
│  ├─ r20_seed1/
│  └─ r20_seed2/
```

This repository provides split definition files under `splits/`, including labeled and unlabeled partitions for all labeled ratios and random seeds used in the paper.

> Note: the DeepPCB dataset itself is not redistributed in this repository.

---

## Main Scripts

### DAT-SCPL

```bash
python run_dat_scpl.py --tag r20_seed0
```

### DAT-only baseline

```bash
python run_dat_only.py --tag r20_seed0
```

### Fixed-threshold SSOD baseline

```bash
python run_fixedtau.py --tag r20_seed0
```

These scripts correspond to the three main settings reported in the paper:

* DAT-SCPL
* DAT-only
* Fixed-threshold SSOD baseline

---

## Reproducing Main Results

### 1. Run experiments

Use the main scripts above for a given ratio/seed tag, for example:

* `r01_seed0`
* `r05_seed1`
* `r10_seed2`
* `r20_seed0`

### 2. Summarize metrics

Utility scripts in `code/` can be used to inspect and summarize outputs, for example:

```bash
python code/print_metrics_r20_seed12.py
python code/print_metrics_r01_r10_seed012.py
```

### 3. Reproduce paper figures and tables

Scripts in `paper_fig_assets/` are used to reproduce paper assets, including:

* main comparison table
* main performance curve
* SCPL sensitivity figure
* pseudo-label statistics

Examples:

```bash
python paper_fig_assets/make_main_table_all_ratios.py
python paper_fig_assets/plot_main_curve_map5095.py
python paper_fig_assets/plot_scpl_sensitivity_paper.py --root YOUR_PROJECT_ROOT --split_tag r10_seed0 --runs_dir runs_ssod_paper3_one --metric mAP50_95
python paper_fig_assets/plot_pseudo_overall.py
```

---

## Notes

* This repository focuses on the **paper reproduction pipeline**, rather than a fully cleaned engineering framework.
* Large model checkpoints, training logs, and dataset files are **not included**.
* Some scripts were originally developed for local Windows-based experiments and may require path adjustment before use in a different environment.

---

## Reproducibility Statement

This repository is intended to support reproducibility of the main experimental results reported in our manuscript submitted to *The Visual Computer*.

It includes:

* split definitions,
* main training and inference scripts,
* baseline scripts,
* metric summarization scripts,
* paper figure generation scripts.

If you use this repository in academic work, please cite the corresponding manuscript.

---
## License

This project is released under the MIT License. See LICENSE for details.
