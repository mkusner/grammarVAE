import sys
#sys.path.insert(0, '..')
import molecule_vae
import numpy as np

# 1. load grammar VAE
grammar_weights = "pretrained/zinc_vae_grammar_L56_E100_val.hdf5"
grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)

# 2. let's encode and decode some example SMILES strings
smiles = ["C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]",
          "CC[NH+](CC)[C@](C)(CC)[C@H](O)c1cscc1Br",
          "O=C(Nc1nc[nH]n1)c1cccnc1Nc1cccc(F)c1",
          "Cc1c(/C=N/c2cc(Br)ccn2)c(O)n2c(nc3ccccc32)c1C#N",
          "CSc1nncn1/N=C\c1cc(Cl)ccc1F"]

# z: encoded latent points
# NOTE: this operation returns the mean of the encoding distribution
# if you would like it to sample from that distribution instead
# replace line 83 in molecule_vae.py with: return self.vae.encoder.predict(one_hot)
z1 = grammar_model.encode(smiles)

# mol: decoded SMILES string
# NOTE: decoding is stochastic so calling this function many
# times for the same latent point will return different answers

for mol,real in zip(grammar_model.decode(z1),smiles):
    print mol + '  ' + real



# 3. the character VAE (https://github.com/maxhodak/keras-molecules)
# works the same way, let's load it
char_weights = "pretrained/zinc_vae_str_L56_E100_val.hdf5"
char_model = molecule_vae.ZincCharacterModel(char_weights)

# 4. encode and decode
z2 = char_model.encode(smiles)
for mol in char_model.decode(z2):
    print mol



