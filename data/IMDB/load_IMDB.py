
import re
import os
import tarfile
import zipfile
from typing import Tuple

import sklearn


# function to read files
def _read_zip_contents(*, imdb_path, pat):
    res = []
    with zipfile.ZipFile(imdb_path) as zfile:
        for filename in zfile.namelist():
            if (not os.path.isdir(filename)) and (pat.match(filename) is not None):
                with zfile.open(filename, 'r') as f:
                    contents = zfile.read(filename)
                    res.append((filename, contents.decode('utf-8')))
    return res


def _read_targz_contents(*, imdb_path, pat):
    res = []
    with tarfile.open(imdb_path, 'r:gz') as tfile:
        for member in tfile.getmembers():
            filename = member.path
            if (not os.path.isdir(filename)) and (pat.match(filename) is not None):
                with tfile.extractfile(member) as f:
                    contents = f.read()
                    res.append((filename, contents.decode('utf-8')))
    return res


# function to read data sets from Imdb directory structure
def _read_d(*, imdb_path, pat):
    # get data
    data = _read_targz_contents(imdb_path=imdb_path, pat=pat)
    texts = [d[1] for d in data]
    paths = [d[0] for d in data]
    pos_pat = re.compile(".*/pos/.*")
    labels = [1 if pos_pat.match(fi) else 0 for fi in paths]
    data = sklearn.utils.Bunch(data=texts, target=labels)
    return data


def load_IMDB_from_zip(imdb_path: str) -> Tuple[sklearn.utils.Bunch, sklearn.utils.Bunch]:
    """
    Load IMDB data from imdb_path zipfile.

    :param imdb_path: zip file path
    :return: tuple of train_data and test data as sklearn.utils.Bunch objects
    """
    # need files that end with ".txt" and don't start with "urls" -- start with numbers
    assert isinstance(imdb_path, str)
    train_pattern = re.compile(r'.*[/\\](train)[/\\]((pos)|(neg))[/\\][0-9_]+\.txt')
    test_pattern = re.compile(r'.*[/\\](test)[/\\]((pos)|(neg))[/\\][0-9_]+\.txt')

    train_data = _read_d(imdb_path=imdb_path, pat=train_pattern)
    test_data = _read_d(imdb_path=imdb_path, pat=test_pattern)
    return train_data, test_data


def load_IMDB() -> Tuple[sklearn.utils.Bunch, sklearn.utils.Bunch]:
    """
    Load IMDB data from aclIMdb.zip in same directory as this code module.

    :return: tuple of train_data and test data as sklearn.utils.Bunch objects
    """
    imdb_code_path = __file__  # find path to this Python source file
    imdb_path = os.path.dirname(imdb_code_path)  # get dir
    imdb_path = os.path.join(imdb_path, 'aclImdb_v1.tar.gz')
    return load_IMDB_from_zip(imdb_path)
