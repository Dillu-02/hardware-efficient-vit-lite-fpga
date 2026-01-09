# Hardware-Efficient Vision Transformer Lite for FPGA

This repository presents the hardware design artifacts and experimental results for a hardware-efficient implementation of the Multi-Head Self-Attention (MHSA) layer used in a lightweight Vision Transformer (ViT-Lite) architecture.

The primary focus of this work is the design and evaluation of an FPGA-friendly MHSA accelerator using High-Level Synthesis (HLS). The design emphasizes memory efficiency, scalable attention computation, and practical deployability on resource-constrained FPGA platforms.

---

## Project Context

Vision Transformers rely on global self-attention to achieve strong performance, but the quadratic complexity and memory requirements of MHSA make direct deployment on edge hardware challenging. This work explores how architectural simplifications and hardware-aware optimizations can make transformer-based attention feasible on FPGA platforms.

The MHSA accelerator implemented in this repository is intended to serve as a standalone, reusable building block within a ViT-Lite pipeline and is evaluated both algorithmically and at the hardware level.

---

## Repository Overview

This repository is intentionally minimal and contains only core design and evaluation artifacts.

- `HLS/` - HLS design files and testbench for the MHSA accelerator
- `data/` - Formatted inputs and exported weights used for HLS validation
- `docs/` - Figures summarizing model and hardware results
- `notebook/` - Reference notebook used for software-side processing and evaluation
- `README.md`


---

## HLS Implementation

The `HLS/` directory contains the C++ implementation of the MHSA accelerator developed using Vitis HLS.

### Contents

- `MHSA.cpp`  
  Top-level HLS design implementing:
  - Q, K, V linear projections
  - Scaled dot-product attention
  - Row-wise softmax normalization
  - Output projection

- `MHSA.h`  
  Header file defining design parameters, tensor dimensions, and interfaces.

- `tb_mhsa.cpp`  
  HLS testbench used for functional verification with formatted inputs and weights.

- `MHSA_HLS.tcl`  
  Automation script used to configure, synthesize, and evaluate the design in Vitis HLS.

### Design Characteristics

- Tile-based matrix multiplication to manage large attention matrices
- Independent processing of attention heads to improve parallelism
- Explicit control over on-chip memory usage to limit BRAM consumption
- Design evaluated at the post-synthesis level for latency and resource utilization

---

## Data Files

The `data/` directory contains formatted input tensors and exported weight files used by the HLS testbench.

These files are generated offline and are included solely to support:
- Functional validation of the MHSA design
- Consistency between software-generated outputs and hardware execution

Raw datasets and trained model checkpoints are intentionally not included.

---

## Software Processing Notebook

The `notebook/` directory contains a reference Jupyter notebook used during model development and evaluation.

The notebook documents:
- Data preprocessing steps
- Model-side evaluation workflow
- Generation of result figures included in the `docs/` directory

The notebook is provided for transparency and contextual understanding of the reported results. It is not intended as a complete end-to-end training or reproduction pipeline.

---

## Experimental Results and Figures

The `docs/` directory contains figures used to evaluate both the ViT-Lite model and the MHSA hardware accelerator.

### Figures Overview

- `tr_and_val_accuracy.png`  
  Training and validation accuracy curves.

- `tr_and_val_loss.png`  
  Training and validation loss curves.

- `confusion_metrics.png`  
  Confusion matrix on the test dataset.

- `parameter_summary_table.png`  
  Layer-wise parameter summary of the ViT-Lite model.

- `dimension_comparison_table.png`  
  Comparison of precision, embedding dimension, and number of attention heads versus resource utilization in the programmable logic (PL).

- `latency_summary.png`  
  Post-synthesis latency summary obtained from Vitis HLS.

---

## FPGA System Integration

An IP-level system integration of the MHSA accelerator was performed using Vivado IP Integrator on a Zynq UltraScale+ MPSoC (ZCU104) platform.

### Vivado Block Design

The Vivado block design illustrates:
- The Zynq UltraScale+ Processing System (PS)
- AXI SmartConnect for high-bandwidth memory access
- The MHSA accelerator instantiated as a custom HLS IP
- Multiple AXI master interfaces for parallel access to input tensors, weights, and outputs
- AXI-Lite control interface for configuration and execution control

This integration demonstrates how the MHSA accelerator can be embedded within a larger hardware–software system while maintaining scalable memory bandwidth and clean separation between control and data paths.

---

## Resource Utilization in PL

Resource utilization refers to the percentage of FPGA programmable logic (PL) resources consumed by the synthesized MHSA accelerator. This includes:

- LUTs and Flip-Flops for control and datapath logic
- BRAM for on-chip buffering of tensors and intermediate results
- DSP slices for multiply–accumulate operations

PL utilization metrics are used to assess scalability, feasibility, and headroom for integration with additional accelerators or system logic.

---

## Notes

- Dataset files and trained model checkpoints are not included.
- FPGA bitstreams and hardware handoff files are intentionally excluded.
- This repository documents design methodology and experimental outcomes rather than providing a turnkey deployment solution.

---

## Summary

This repository demonstrates that transformer-style self-attention can be implemented efficiently on FPGA platforms through careful architectural choices and hardware-aware optimization. The presented MHSA accelerator provides a practical foundation for deploying ViT-Lite models in edge and embedded scenarios where latency and on-chip memory are critical constraints.
