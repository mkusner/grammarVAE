
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

import sascorer

import numpy as np  

# We load the smiles data

fname = '../../data/250k_rndm_zinc_drugs_clean.smi'

with open(fname) as f:
    smiles = f.readlines()

for i in range(len(smiles)):
    smiles[ i ] = smiles[ i ].strip()

# We load the auto-encoder

import sys
sys.path.insert(0, '../../')
import molecule_vae
char_weights = "../../pretrained/zinc_vae_str_L56_E100_val.hdf5"
char_model = molecule_vae.ZincCharacterModel(char_weights)

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw
import image
import copy
import time

smiles_rdkit = []
for i in range(len(smiles)):
    smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[ i ])))
    print(i)

logP_values = []
for i in range(len(smiles)):
    logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))
    print(i)

SA_scores = []
for i in range(len(smiles)):
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[ i ])))
    print(i)

import networkx as nx

cycle_scores = []
for i in range(len(smiles)):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)
    print(i)

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

latent_points = char_model.encode(smiles_rdkit)

# We store the results

latent_points = np.array(latent_points)
np.savetxt('latent_faetures.txt', latent_points)

targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
np.savetxt('targets.txt', targets)
np.savetxt('logP_values.txt', np.array(logP_values))
np.savetxt('SA_scores.txt', np.array(SA_scores))
np.savetxt('cycle_scores.txt', np.array(cycle_scores))
