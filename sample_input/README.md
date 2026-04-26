# 📡 `sample_input/`: Test Data & Generators

To prove that the model works, you need data to feed it! Since raw Sentinel-1 SAR tiles are huge (often hundreds of megabytes), we created this directory so you can easily generate or download small test samples to run the model locally.

## What's inside?

### `generate_synthetic.py`
If you don't want to download any real satellite imagery, just run this script! It uses `numpy` to mathematically generate a completely fake, but structurally accurate, SAR tile (`sample_sar.npy`). 
- It simulates the VV and VH bands of radar backscatter.
- It's super small (~2MB) and runs in milliseconds.
- Run it with: `python sample_input/generate_synthetic.py`

### `download_sample.py`
If you want to see how the model performs on the *real deal*, run this script.
- It reaches out to a public Google Cloud Storage bucket and pulls down an actual test tile from the Sen1Floods11 dataset (specifically, a tile over Sri Lanka).
- Run it with: `python sample_input/download_sample.py`

### Data Files
When you run the scripts above, you'll see files like `sample_sar.npy` and `Sri-Lanka_534068.tif` pop up in this folder. You can pass these file paths directly to `infer.py` to test the model.
