# Grammar Variational Autoencoder

This repository contains training and sampling code for the paper: <a href="https://arxiv.org/abs/1703.01925">Grammar Variational Autoencoder</a>.


## Requirements

Install (CPU version) using `pip install -r requirements.txt`

For GPU compatibility, replace the fourth line in requirements.txt with: https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl


## Creating datasets

### Molecules

To create the molecule datasets, call:

* `python make_zinc_dataset_grammar.py`
* `python make_zinc_dataset_str.py`

### Equations

The equation dataset can be downloaded here: [grammar](https://www.dropbox.com/s/yq1gpygw3oq1grq/eq2_grammar_dataset.h5?dl=0), [string](https://www.dropbox.com/s/gn3iq2ykrs0dqwb/eq2_str_dataset.h5?dl=0)


## Training

### Molecules

To train the molecule models, call:

* `python train_zinc.py` % the grammar model
* `python train_zinc.py --latent_dim=2 --epochs=50` % train a model with a 2D latent space and 50 epochs 
* `python train_zinc_str.py`

### Equations

* `python train_eq.py` % the grammar model
* `python train_eq.py --latent_dim=2 --epochs=50` % train a model with a 2D latent space and 50 epochs 
* `python train_eq_str.py`


## Sampling

### Molecules

The file molecule_vae.py can be used to encode and decode SMILES strings. For a demo run:

* `python encode_decode_zinc.py`

### Equations

The analogous file equation_vae.py can encode and decode equation strings. Run:

* `python encode_decode_eq.py`
