from pandas import DataFrame, SparseDataFrame, Series
from scipy.special import digamma
from scipy.sparse import csr_matrix
from numpy import float32


def parse_data(data, data_formats):
    """ Function to parse 2D list data """
    num_rows = len(data)

    for row_id in range(num_rows):  # For each row in data
        for col_id, formatter in enumerate(data_formats):   # For each column in row of data
            if formatter is not None:
                data[row_id][col_id] = formatter(data[row_id][col_id])


def print_data(data):
    """ Debugging function to print 2D list """
    for row in data:
        print(row)


def split_sets(dataset, splitter, test_frac=.2, labels=None):
    """
    Splits the dataset into a training and test set.
    :param dataset: Full dataset, dataframe
    :param splitter: Accessor function, dataframe -> (feature_cols, label_cols)
    :param test_frac: Fraction of dataset to be used as the test set, (default 20%)
    :return: (train_feat, train_label), (test_feat, test_labels)
    """

    if type(dataset) is not DataFrame and type(dataset) is not SparseDataFrame:
        raise TypeError('Dataset must be a (Pandas) DataFrame')
    if test_frac < 0 or test_frac > 1:
        raise ValueError('test_frac is out of range, must be in [0, 1]')
    if labels is not None and type(labels) is not Series:
        raise TypeError('Labels must be a (Pandas) Series')

    num_rows = len(dataset.index)
    pivot_index = int(num_rows * (1 - test_frac))

    train_set = dataset.iloc[:pivot_index, :]
    test_set = dataset.iloc[pivot_index:, :]

    if labels is not None:
        train_label = labels.iloc[:pivot_index]
        test_label = labels.iloc[pivot_index:]

        return (splitter(train_set), splitter(test_set)), (train_label, test_label)

    return splitter(train_set), splitter(test_set)


# TODO double check function
def normalize_doc_term(dataset):
    """ Normalizes a document-term matrix using the di-gamma function """
    if type(dataset) is not DataFrame:
        return TypeError('Dataset must be a (Pandas) Dataframe')

    return dataset.applymap(digamma)


def to_csr_matrix(dataset, conv_type=float32):
    """ Takes a (most likely sparse) dataset and converts it to a Scipy CSR matrix, necessary for XGBoost """
    set_type = type(dataset)
    if set_type is not DataFrame and set_type is not SparseDataFrame:
        raise TypeError('Dataset must be a (Pandas) [Sparse]DataFrame')

    sparse_dataset = csr_matrix(dataset.astype(conv_type).to_coo())

    return sparse_dataset
