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
