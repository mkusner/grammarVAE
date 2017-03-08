import nltk
import re

import eq_grammar
import molecule_vae
import models.model_eq
import models.model_eq_str
import numpy as np


def tokenize(s):
    funcs = ['sin', 'exp']
    for fn in funcs: s = s.replace(fn+'(', fn+' ')
    s = re.sub(r'([^a-z ])', r' \1 ', s)
    for fn in funcs: s = s.replace(fn, fn+'(')
    return s.split()

class EquationGrammarModel(molecule_vae.ZincGrammarModel):
    
    def __init__(self, weights_file, latent_rep_size=25):
        """ Load the (trained) equation encoder/decoder, grammar model. """
        self._grammar = eq_grammar
        self._model = models.model_eq
        self.MAX_LEN = 15 # TODO: read from elsewhere
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = tokenize
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix
        self.vae = self._model.MoleculeVAE()
        self.vae.load(self._productions, weights_file, max_length=self.MAX_LEN, latent_rep_size=latent_rep_size)


class EquationCharacterModel(object):

    def __init__(self, weights_file, latent_rep_size=25):
        self._model = models.model_eq_str
        self.MAX_LEN = 19
        self.vae = self._model.MoleculeVAE()
        self.charlist = ['x', '+', '(', ')', '1', '2', '3', '*', '/', 's', 'i', 'n', 'e', 'p', ' ']
        #self.charlist = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
        #                 '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
        #                 '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
        self.vae.load(self.charlist, weights_file, max_length=self.MAX_LEN, latent_rep_size=latent_rep_size)

    def encode(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in xrange(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        return self.vae.encoderMV.predict(one_hot)[0]

    def decode(self, z):
        """ Sample from the character decoder """
        assert z.ndim == 2
        out = self.vae.decoder.predict(z)
        noise = np.random.gumbel(size=out.shape)
        sampled_chars = np.argmax(np.log(out) + noise, axis=-1)
        char_matrix = np.array(self.charlist)[np.array(sampled_chars, dtype=int)]
        return [''.join(ch).strip() for ch in char_matrix]
