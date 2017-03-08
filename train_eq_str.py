from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from models.model_eq_str import MoleculeVAE
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import h5py

MAX_LEN = 19
DIM = 15
LATENT = 10
EPOCHS = 50
BATCH = 600

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()

charset = ['x', '+', '(', ')', '1', '2', '3', '*', '/', 's', 'i', 'n', 'e', 'p', ' ']
def main():

    # 0. load dataset
    h5f = h5py.File('data/eq2_str_dataset.h5', 'r')
    data = h5f['data'][:]
    h5f.close()

    # 1. get any arguments and define save file, then create the VAE model
    args = get_arguments()
    params = {'hidden': 200, 'dense': 200, 'conv1': 2, 'conv2': 3, 'conv3': 4}
    model_save = 'eq_vae_str_h' + str(params['hidden']) + '_c123_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '.hdf5'
    model = MoleculeVAE()

    # 2. if this results file exists already load it
    if os.path.isfile(model_save):
        model.load(charset, model_save, latent_rep_size = args.latent_dim, hypers = params)
    else:
        model.create(charset, max_length=MAX_LEN, latent_rep_size = args.latent_dim, hypers = params)

    # 3. only save best model found on a 10% validation set
    checkpointer = ModelCheckpoint(filepath = model_save,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 1,
                                  min_lr = 0.0001)

    # 4. fit the vae
    model.autoencoder.fit(
        data,
        data,
        shuffle = True,
        nb_epoch = args.epochs,
        batch_size = BATCH,
        callbacks = [checkpointer, reduce_lr],
        validation_split = 0.1
    )

if __name__ == '__main__':
    main()
