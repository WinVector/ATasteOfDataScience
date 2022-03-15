
from typing import Any, Dict, Tuple
import numpy

from typing import Any, Dict, Tuple
import os
import gzip
import pickle
import pandas
import tensorflow.keras as keras
import hashlib
from GloVe_tools import glove_iterator


def load_coding_bundle(train_text, *, glove_path: str) -> Tuple[Any, Dict]:
    """
    Build or load coding bundle.

    :param train_text: texts to train tokenizer on
    :param glove_path: path to GloVe word embeddings
    :return: Tuple: tokenizer, token id to vector dictionary
    """
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000)  # using a small vocabulary to save space
    tokenizer.fit_on_texts(train_text)
    words = [tokenizer.index_word[i + 1] for i in range(len(tokenizer.index_word))]
    hash_key = hashlib.sha256(bytes(' '.join(words), 'utf-8')).hexdigest()
    cache_file = f'glove_cache_{hash_key}.pkl.gz'
    if not os.path.exists(cache_file):
        code_book = dict()
        for key, codes in glove_iterator(glove_path):
            enc = tokenizer.texts_to_sequences([key])[0]
            if len(enc) > 0:
                for tok in enc:
                    if tok not in code_book.keys():
                        code_book[tok] = codes
        coding_bundle = {'tokenizer': tokenizer, 'code_book': code_book}
        with gzip.open(cache_file, 'wb') as cache_stream:
            pickle.dump(coding_bundle, cache_stream)
    else:
        with gzip.open(cache_file, 'rb') as cache_stream:
            coding_bundle = pickle.load(cache_stream)
    return coding_bundle['tokenizer'], coding_bundle['code_book']


class GloveEncoder():
    """Class to use glove encodings to convert words to vectors"""
    def __init__(
            self,
            *,
            glove_path: str,
            k: int = 2,
    ):
        assert isinstance(glove_path, str)
        assert isinstance(k, int)
        assert k > 0
        self.glove_path = glove_path
        self.k = k
        self.tokenizer = None  # maps strings to tokens
        self.code_book = None  # maps tokens to vectors
        self.word_embed_dim = None  # not known yet
        self.encode_dim = None  # not known yet

    def fit(self, X, y=None):
        tokenizer, code_book = load_coding_bundle(train_text=X, glove_path=self.glove_path)
        self.tokenizer = tokenizer
        self.code_book = code_book
        k0, v0 = next(iter(self.code_book.items()))
        self.word_embed_dim = len(v0)
        self.encode_dim = len(self._encode_from_vecs([]))
        return self
    
    def _encode_from_vecs(self, vecs):
        k = self.k
        if len(vecs) < 1:  # missed all tokens
            vecs = [numpy.zeros(self.word_embed_dim)]
        if len(vecs) < k:
            vecs = vecs + [vecs[len(vecs)-1]] * (k - len(vecs))
        # build k-grams
        k_grams = [numpy.concatenate(
            [vecs[i + d] for d in range(k)]
        ) for i in range(len(vecs) - (k - 1))]
        assert len(k_grams) > 0   # should be true by earlier guards, easier to catch here
        return numpy.average(numpy.asarray(k_grams), axis=0)  # bag of codes model

    def encode_text(self, txt: str):
        k = self.k
        assert isinstance(k, int)
        assert k > 0
        toks = self.tokenizer.texts_to_sequences([txt])[0]
        vecs = []
        for t in toks:
            try:
                vec = self.code_book[t]
                vecs.append(vec)
            except KeyError:
                pass
        return self._encode_from_vecs(vecs)

    def transform(self, X):
        rows = [self.encode_text(txt) for txt in X]
        frame = pandas.DataFrame(rows)
        return frame
