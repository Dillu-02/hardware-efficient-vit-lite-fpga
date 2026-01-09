# Hardware-Efficient Vision Transformer Lite for FPGA

This repository contains the hardware design artifacts and experimental results for a hardware-efficient implementation of the Multi-Head Self-Attention (MHSA) layer used in a lightweight Vision Transformer (ViT-Lite) architecture.

The focus of this work is on designing an FPGA-friendly MHSA accelerator using High-Level Synthesis (HLS), with careful attention to memory usage, latency, and scalability for edge deployment scenarios.

---

## Repository Overview

This repository is intentionally minimal and contains only the core design and evaluation artifacts.
- `HLS/` - HLS implementation of the MHSA accelerator
- `docs/` - Experimental results and evaluation figures

---

## HLS Implementation

The `HLS/` directory contains the C++ implementation of the MHSA accelerator written for Vitis HLS.

### Contents
- `MHSA.cpp`  
  Top-level HLS design implementing:
  - Q, K, V linear projections
  - Scaled dot-product attention
  - Row-wise softmax normalization
  - Output projection

- `MHSA_HLS.tcl`  
  HLS automation script used for synthesis and evaluation.

The design follows a tile-based computation strategy and processes attention heads independently to improve throughput while controlling BRAM utilization.

---

## Experimental Results

The `docs/` directory contains figures used to evaluate both model performance and hardware efficiency.

### Included Results
- Training and validation accuracy
- Training and validation loss
- Confusion matrix on test data
- Parameter summary of the ViT-Lite model
- Latency summary from HLS synthesis
- Comparison of embedding and head dimensions vs resource usage

These results collectively demonstrate the feasibility of deploying transformer-based attention mechanisms on FPGA platforms under resource constraints.

---

## Notes

- Dataset files and trained model checkpoints are not included.
- FPGA bitstreams and hardware handoff files are intentionally excluded.
- This repository is intended to document the design methodology and experimental outcomes rather than provide a turnkey deployment package.

