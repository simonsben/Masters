from numpy.random import randint
from scipy.sparse import csr_matrix
from utilities.data_management import split_sets


def test_split():
    mat = csr_matrix(randint(0, 2, (10, 5)))
    labels = randint(0, 1, 10)
    (train, test), (train_l, test_l) = split_sets(mat, labels=labels)

    assert train.shape[0] == 7, 'Train matrix is wrong length'
    assert train.shape[1] == 5, 'Train matrix lost column(s)'
    assert test.shape[0] == 3, 'Test matrix is wrong length'
    assert test.shape[1] == 5, 'Test matrix lost column(s)'

    assert len(train_l.shape) == 1, 'Train labels are not a matrix'
    assert len(test_l.shape) == 1, 'Test labels are not a matrix'
    assert train_l.shape[0] == 7, 'Train labels are wrong length'
    assert test_l.shape[0] == 3, 'Test labels are wrong length'


def test_prepare():
    mat = csr_matrix(randint(0, 2, (10, 5)))
    labels = randint(0, 1, 10)
