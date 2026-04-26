# 🧠 `src/`: The Inference Engine

Welcome to the brains of the operation! This directory contains the core machine learning logic required to take our fine-tuned IBM TerraMind checkpoint and use it to predict floods.

By separating this code from the API and the CLI scripts, we keep everything modular and reusable. Whether you're running a massive batch job via the command line or serving requests through a REST endpoint, all roads lead here.

## What's inside?

### `model.py`
This is where the magic happens. 
- It uses `terratorch` to load the saved `.ckpt` weights into the `SemanticSegmentationTask` architecture.
- It automatically detects your hardware (using a GPU if you have one, or falling back to your CPU if you don't) to run inference as fast as possible.
- It contains the `run_inference` function, which takes raw satellite data, passes it through the model, and calculates all the cool telemetry metrics (like how many square kilometers of land are flooded, how long the inference took, and how much bandwidth was saved).

### `preprocess.py`
Before a neural network can look at a satellite image, the image needs to be cleaned up. SAR (Synthetic Aperture Radar) data can be messy—it often contains NaN (Not a Number) values or infinite values, and it needs to be cast to the correct data types (like `float32`).
- This script provides helper functions to sanitize and normalize the incoming arrays so they are perfectly formatted for the TerraMind backbone.
