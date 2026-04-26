# 🌊 FloodSense: Intelligent SAR Flood Detection

Welcome to **FloodSense**! This project tackles one of the most pressing challenges in disaster response: accurately and rapidly identifying flooded areas from space. 

When a flood hits, bad weather and cloud cover usually ground optical satellites. That's why we use **Sentinel-1 SAR (Synthetic Aperture Radar)** imagery—it can see right through the clouds, day or night. To process this complex data, we've fine-tuned IBM's **TerraMind v1 Tiny** foundation model on the **Sen1Floods11** dataset.

The result? A lightweight, edge-ready AI model that achieves a **0.78 mIoU** on hand-labeled test data, capable of quickly segmenting flood from dry land.

---

## 🚀 Quickstart Guide for Judges (and curious developers!)

We've designed this to be as easy to run as possible.

### 1. Set up your environment
We recommend using Python 3.11 or 3.12 (3.13 isn't supported by some ML dependencies yet).

```bash
# Optional but recommended: create a virtual environment
python -m venv .venv
# Activate it (Windows)
.venv\Scripts\activate
# Activate it (Mac/Linux)
# source .venv/bin/activate

# Install the necessary dependencies
pip install -r requirements.txt
```

### 2. Get some sample data
You can generate a tiny synthetic demo tile right away, or download a real piece of Sentinel-1 imagery:
```bash
# Generate a quick 2MB synthetic tile to test the pipeline:
python sample_input/generate_synthetic.py

# OR download a real Sri Lanka flood tile from Sen1Floods11:
python sample_input/download_sample.py
```

### 3. Run the Inference!
Use our single CLI entry point to see the model in action:
```bash
python infer.py "backend/mIoU=0.78.ckpt" "sample_input/sample_sar.npy"
```

You should see a beautifully detailed "Disaster Assessment Report" pop up in your console, showing the exact model telemetry, the time it took, and the estimated flooded area.

---

## 📁 What's in the repo?

We've broken down the project to make it modular and easy to understand. Check out the individual READMEs inside these folders for more details:

* **[`src/`](src/README.md)**: The heart of the inference engine. This is where the TerraMind checkpoint is loaded and the math happens.
* **[`api/`](api/README.md)**: A FastAPI REST service that wraps our model, ready to be consumed by a frontend or mobile app.
* **[`sample_input/`](sample_input/README.md)**: Scripts and data for testing the model without needing a massive dataset.
* **[`backend/`](backend/README.md)**: Holds the fine-tuned model checkpoint (`mIoU=0.78.ckpt`).
* **[`configs/`](configs/README.md)**: Configuration files used during the fine-tuning process.
* **[`notebooks/`](notebooks/README.md)**: Contains a link to our training pipeline.
* **`infer.py`**: The main command-line interface to test the model.
* **`418Hackathon_fixed.ipynb`**: The actual Google Colab notebook we used to train and fine-tune the model.

---

## 🧠 Under the Hood

Curious about the ML details? Here's a quick cheat sheet on how we built this:

| Feature | What we used |
|---|---|
| **Foundation model** | IBM TerraMind v1 Tiny |
| **Architecture** | UperNet Decoder |
| **Input Data** | Sentinel-1 GRD (VV + VH bands) |
| **Output** | Binary mask (0 = land, 1 = flood) |
| **Training Dataset** | Sen1Floods11 (hand-labeled split) |
| **Frameworks** | `terratorch ≥ 1.2.4` + `PyTorch Lightning` |

We trained this for 50 epochs using mixed precision (AMP) and a combination of CrossEntropy and Dice loss, stopping early when the model peaked at an **mIoU of 0.78**. 

---

Thanks for checking out FloodSense! If you have any issues running the code, make sure your Python version is compatible and that you've installed all requirements.
