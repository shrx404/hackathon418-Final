# 🧠 `src/`: The Inference Engine

This directory contains the core machine learning logic required to take the fine-tuned checkpoint and predict floods.

## What's inside?

- **`__init__.py`**: Makes the `src` directory an importable Python module.
- **`model.py`**: Loads the `.ckpt` weights into the model architecture, manages hardware acceleration (GPU/CPU), and provides the `run_inference` function to execute predictions and calculate telemetry.
- **`preprocess.py`**: Contains helper functions to sanitize and normalize incoming satellite data arrays (e.g., handling NaNs, type casting) so they are perfectly formatted for the model backbone.
