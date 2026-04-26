# 🗄️ `backend/`: Model Checkpoints

This folder acts as our local storage for heavy, trained model weights. 

Currently, it houses our prized possession: **`mIoU=0.78.ckpt`**.

## About the checkpoint

This file contains the PyTorch Lightning state dictionaries, hyperparameters, and specific TerraMind architectures that we fine-tuned. It's about 150MB in size. 

When you run `infer.py` or start the FastAPI server, the code points to this file to load the neural network's "brain" into memory.

*Note: In a massive production environment, you might pull these weights from an S3 bucket or an MLflow registry at runtime, but keeping it here locally makes it incredibly easy for judges and developers to test the project right out of the box!*
