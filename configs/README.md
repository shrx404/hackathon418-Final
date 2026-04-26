# Settings for Training and Model

The `configs/` folder has all the files that we use to set up and adjust the model.

## What is in this folder?

- **`terramind_seg.yaml`**: This is a file that has all the details for the network. It tells us about:

The model we are using (`terramind_v1_tiny` and others)
The decoder settings we need (`UperNet`)
How fast we learn, how many things we look at each time and what we do when we are wrong
Where to find the data and how to split it up

The `configs/` folder is where we keep all the model configuration files. We use these files to set up the model and make it work right.
The `terramind_seg.yaml` file is like a recipe, for the network. It has all the details we need to make the model work.
