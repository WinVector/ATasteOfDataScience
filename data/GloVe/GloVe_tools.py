
from typing import Any, Dict, Tuple
import zipfile
import numpy

from sklearn.neighbors import NearestNeighbors


def glove_iterator(glove_path: str):
    assert isinstance(glove_path, str)
    expected_len = None
    with zipfile.ZipFile(glove_path) as zfile:
        for filename in zfile.namelist():
            if filename.endswith('.txt'):  # assume only .txt file is the one we want
                with zfile.open(filename, 'r') as glove_lines:
                    while line := glove_lines.readline():
                        line = line.decode('utf-8').strip()
                        if len(line) > 0:
                            toks = line.split(' ')
                            if len(toks) > 1:
                                key = toks[0]
                                if len(key) > 0:
                                    codes = numpy.asarray(toks[1:], dtype=float)
                                    len_codes = len(codes)
                                    if len_codes > 0:
                                        if expected_len is None:
                                            expected_len = len_codes
                                        if expected_len == len_codes:
                                            yield key, codes


def prune_nbhd(nbhd, k: int):
    # prune to approximate nbhd
    n = len(nbhd)
    if n <= k:
        return nbhd.copy()
    values = [v for v in nbhd.values()]
    values.sort()
    target = values[k-1]
    keep = {k: v for k, v in nbhd.items() if v <= target}
    return keep


def scan_for_knn(*, glove_path: str, vec_key: str, k: int = 5) -> Tuple[Any, Dict]:
    """
    Lookup GloVe code vector and nearest neighbors.

    :param glove_path: path to the Glove data
    :param vec_key: string to lookup
    :param k: number of neighbors
    :return: tuple of vector and neighbor strings to distance squared dictionary
    """
    assert isinstance(glove_path, str)
    assert isinstance(vec_key, str)
    assert isinstance(k, int)
    assert k > 0
    found_code = None
    for key, codes in glove_iterator(glove_path):
        if key == vec_key:
            found_code = codes
            break
    if found_code is None:
        raise KeyError(f'could not find "{vec_key}"')
    nbhd = dict()
    for key, codes in glove_iterator(glove_path):
        if key != vec_key:
            dist_sq = numpy.sum((found_code - codes)**2)
            nbhd[key] = dist_sq
            if len(nbhd) > 10*k:
                nbhd = prune_nbhd(nbhd, k)
    return found_code, prune_nbhd(nbhd, k)


class GloveKNN:
    def __init__(self, *, glove_path: str, k: int = 5):
        assert isinstance(glove_path, str)
        assert isinstance(k, int)
        assert k > 0
        self.glove_path = glove_path
        words = []
        vectors = []
        self.kmap = dict()
        for key, codes in glove_iterator(glove_path):
            words.append(key)
            vectors.append(codes)
            self.kmap[key] = codes
        self.words = numpy.array(words)
        vectors = numpy.array(vectors)
        self.nbrs = NearestNeighbors(n_neighbors=k).fit(vectors)

    def kneighbors_v(self, X):
        """lookup neighbors for rows of X"""
        distances, indices = self.nbrs.kneighbors(X)
        return [[self.words[nbrs[j]] for j in range(len(nbrs))] for nbrs in indices]

    def kneighbors_k(self, key: str):
        """lookup neighbors for string key"""
        assert isinstance(key, str)
        vec = self.kmap[key]
        return self.kneighbors_v([vec])[0]
