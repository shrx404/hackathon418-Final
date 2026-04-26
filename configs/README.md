# ⚙️ `configs/`: Training & Model Configurations

This folder stores the configuration files we used when originally setting up and fine-tuning the model. 

## What's inside?

### `terramind_seg.yaml`
This is a `terratorch` configuration file. Think of it as a recipe book for how the neural network should be built and trained.

Inside you'll find parameters defining:
- The base foundation model (`terramind_v1_tiny`)
- The decoder used to make sense of the features (`UperNet`)
- Learning rates, batch sizes, and loss functions (like Dice + CrossEntropy)
- The dataset paths and splits

While you don't technically *need* this file to just run inference using `infer.py` (since those details are saved inside the `.ckpt` file), we keep it here for transparency and documentation, so anyone can see exactly how we configured our training run.
