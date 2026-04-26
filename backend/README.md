# 🗄️ `backend/`: Model Checkpoints

This folder acts as local storage for heavy, trained model weights. 

## What's inside?

- **`mIoU=0.78.ckpt`**: The PyTorch Lightning state dictionaries, hyperparameters, and TerraMind architecture weights fine-tuned for our task. It represents a model that achieved a Mean Intersection over Union of 0.78. When the model is loaded, it pulls its brain from this file.
