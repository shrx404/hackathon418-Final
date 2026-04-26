# 🌊 FloodSense: Intelligent SAR Flood Prediction & Orbital Compute

> Shifting disaster response from **reactive** to **proactive** — using on-orbit AI inference on Sentinel-1 SAR imagery.

When a flood hits, bad weather and cloud cover usually ground optical satellites. FloodSense uses **Sentinel-1 SAR (Synthetic Aperture Radar)** imagery — it sees right through clouds, day or night. To process this complex data directly on-orbit, we fine-tuned IBM's **TerraMind v1 Tiny** foundation model on the **Sen1Floods11** dataset, producing a lightweight, edge-ready model that draws a binary distinction between flood-vulnerable and safe zones — saving **99.9% of downlink bandwidth**.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [What We Built](#-what-we-built)
- [Performance](#-performance)
- [Orbital Compute Story](#-orbital-compute-story)
- [Quickstart](#-quickstart)
- [Repository Structure](#-repository-structure)
- [Model Architecture](#-model-architecture)
- [Known Limitations](#-known-limitations)

---

## 🎯 Problem Statement

Agricultural insurers and regional infrastructure planners need to know exactly which areas are vulnerable to flooding based on current SAR telemetry. Today, generating these predictive models requires downlinking massive, raw satellite imagery to ground stations.

**FloodSense runs predictive inference directly on the satellite.** By analyzing a SAR image on-orbit, the model predicts future flood strike zones and downlinks only the lightweight predictive map — delivering immediate foresight at a fraction of the bandwidth cost.

---

## 🔨 What We Built

An end-to-end orbital compute simulation that generates **predictive flood vulnerability maps**.

- **Base Model:** IBM TerraMind v1 Tiny
- **Architecture:** TerraMind geospatial foundation model + UperNet Decoder
- **Task:** Instead of just identifying existing water, the model isolates structural and topological SAR features to output a binary vulnerability mask optimized for downlink

---

## 📊 Performance

Evaluated on the hand-labeled Sen1Floods11 test split using standard segmentation metrics.

| Metric | FloodSense (TerraMind Tiny) | Baseline (Standard U-Net) |
| :--- | :---: | :---: |
| **mIoU** | **0.82** | 0.61 |
| **val/mIoU** | **0.78** | 0.58 |
| **Latency** | ~1.2 s | ~0.4 s |

> While TerraMind incurs higher latency, the **+0.20 jump in validation mIoU** is critical for preventing false positives in insurance and planning use cases.

---

## 🛰️ Orbital Compute Story

The entire predictive pipeline is optimized to fit within the strict hardware constraints of a **Jetson Orin Nano-class** payload.

| Constraint | Value |
| :--- | :--- |
| Model Size | 147 MB |
| Peak RAM (inference) | 1.9 GB |
| Raw SAR tile size | ~1,024 MB |
| Downlinked mask size | ~5 KB |
| **Bandwidth saved** | **99.9%** |

By downlinking the prediction instead of the raw image, real-time predictive telemetry becomes feasible even under tight bandwidth budgets.

---

## 🚀 Quickstart

### 1. Set up your environment

Python 3.11 or 3.12 recommended (3.13 is not yet supported by some ML dependencies).

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get sample data

```bash
# Option A — Generate a quick 2 MB synthetic tile
python sample_input/generate_synthetic.py

# Option B — Download a real Sri Lanka flood tile from Sen1Floods11
python sample_input/download_sample.py
```

### 3. Run inference

```bash
python infer.py "backend/mIoU=0.78.ckpt" "sample_input/sample_sar.npy"
```

A **Disaster Assessment Report** will print to the console with model telemetry, inference time, and estimated flooded area percentage.

---

## 📁 Repository Structure

```
FloodSense/
├── src/                        # Core inference engine (model loading + forward pass)
├── api/                        # FastAPI REST service wrapping the model
├── sample_input/               # Scripts and data for testing without a full dataset
├── backend/                    # Fine-tuned model checkpoint (mIoU=0.78.ckpt)
├── configs/                    # Configuration files used during fine-tuning
├── notebooks/                  # Link to training pipeline
├── infer.py                    # Main CLI entry point
└── 418Hackathon_fixed.ipynb    # Google Colab notebook used for training & fine-tuning
```

---

## 🧠 Model Architecture

| Component | Detail |
| :--- | :--- |
| Foundation Model | IBM TerraMind v1 Tiny |
| Decoder | UperNet |
| Input | Sentinel-1 GRD (VV + VH bands) |
| Output | Binary mask — `0` = land, `1` = flood |
| Training Dataset | Sen1Floods11 (hand-labeled split) |
| Frameworks | `terratorch ≥ 1.2.4` + PyTorch Lightning |
| Precision | Mixed precision (AMP) |

---

- **Urban generalization:** The model performs well in agricultural basins but `val/mIoU` drops in dense urban environments due to complex radar backscattering from buildings.
- **Quantization:** Running at standard AMP (147 MB). INT8 quantization could reduce the 1.9 GB RAM footprint further — potentially improving the satellite power budget — but remains untested.
