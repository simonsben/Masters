from empath import Empath
from pandas import DataFrame
from scipy.sparse import coo_matrix, vstack
from numpy import zeros
from multiprocessing import Pool
import config

# Initialize empath lexicon
lexicon = Empath()

# Get empath features
features = lexicon.analyze('')
num_features = len(features)


def compute_statistic(document):
    tmp = lexicon.analyze(document) if isinstance(document, str) else None
    tmp = coo_matrix(list(tmp.values())) if tmp is not None else coo_matrix(zeros(num_features))

    return tmp


def empath_matrix(dataset):
    """ Computes document empath-feature matrix for dataset """
    if not isinstance(dataset, DataFrame):
        raise TypeError('Dataset must be a (Pandas) DataFrame')

    n_threads = config.n_threads
    workers = Pool(n_threads)

    document_matrix = workers.map(compute_statistic, dataset['document_content'].values)
    document_matrix = vstack(document_matrix).tocsr()

    return document_matrix, features
