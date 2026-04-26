# 📡 `sample_input/`: Test Information and Tools

This folder has some examples and tools to help you get and make test information for the model.

## What is in this folder?

- **`download_sample.py`**: This script gets a test piece, from a Google Cloud Storage bucket. For example it can get something from the Sen1Floods11 dataset.

- **`generate_synthetic.py`**: This script makes a test piece that looks like a real one. It uses numpy. Is useful when you do not want to download big datasets.

- **`sample_sar.npy`**: This is a test SAR data file. You can use it to test the model.
