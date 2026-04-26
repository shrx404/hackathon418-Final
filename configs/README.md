# ⚙️ `configs/`: Training & Model Configurations

This folder stores the configuration files used for setting up and fine-tuning the model. 

## What's inside?

- **`terramind_seg.yaml`**: A `terratorch` configuration file containing the recipe for the neural network. It includes:
  - Base foundation model details (`terramind_v1_tiny` etc.)
  - Decoder settings (`UperNet`)
  - Learning rates, batch sizes, and loss functions
  - Dataset paths and splits
