# Notebooks

| Notebook | Purpose |
|---|---|
| [`418Hackathon_fixed.ipynb`](../418Hackathon_fixed.ipynb) | Full training pipeline — data download, model definition, fine-tuning, evaluation, and inference visualization. |

## Running the Training Notebook

The notebook is designed for **Google Colab with GPU** (T4 or A100).

1. Open `418Hackathon_fixed.ipynb` in Colab.
2. Set runtime to **GPU**.
3. Run all cells top to bottom.
4. The best checkpoint is saved to `checkpoints/terramind-sar-flood-*.ckpt`.

The final checkpoint (`mIoU=0.78`) is already included in `backend/mIoU=0.78.ckpt`.
