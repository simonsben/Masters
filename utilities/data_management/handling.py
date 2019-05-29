from pandas import DataFrame, SparseDataFrame, Series
from scipy.special import digamma
from scipy.sparse import csr_matrix
from numpy import float64, array
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder



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


def split_sets(dataset, splitter, test_frac=.3, labels=None):
    """
    Splits the dataset into a training and test set.
    :param dataset: Full dataset, dataframe
    :param splitter: Accessor function, dataframe -> (feature_cols, label_cols)
    :param test_frac: Fraction of dataset to be used as the test set, (default 20%)
    :param labels: Data labels for dataset, (default None)
    :return: (train_feat, train_label), (test_feat, test_labels)
    """

    if type(dataset) not in [DataFrame, SparseDataFrame, Series]:
        raise TypeError('Dataset must be a (Pandas) DataFrame')
    if test_frac < 0 or test_frac > 1:
        raise ValueError('test_frac is out of range, must be in [0, 1]')
    if labels is not None and type(labels) is not Series:
        raise TypeError('Labels must be a (Pandas) Series')

    num_rows = len(dataset.index)
    pivot_index = int(num_rows * (1 - test_frac))

    if type(dataset) is Series:
        train_set, test_set = dataset.iloc[:pivot_index], dataset.iloc[pivot_index:]
    else:
        train_set, test_set = dataset.iloc[:pivot_index, :], dataset.iloc[pivot_index:, :]

    if labels is not None:
        train_label = labels.iloc[:pivot_index]
        test_label = labels.iloc[pivot_index:]

        return (splitter(train_set), splitter(test_set)), (train_label, test_label)

    return splitter(train_set), splitter(test_set)


# TODO double check function
def normalize_doc_term(dataset):
    """ Normalizes a document-term matrix using the di-gamma function """
    if type(dataset) is not list:
        dataset = [dataset]

    for matrix in dataset:
        if type(matrix) is not csr_matrix:
            return TypeError('Dataset must be a CSR matrix (sparse)')

        matrix.data = digamma(matrix.data)


def to_csr_matrix(dataset, conv_type=float64):
    """ Takes a (most likely sparse) dataset and converts it to a Scipy CSR matrix, necessary for XGBoost """
    set_type = type(dataset)
    if set_type is not DataFrame and set_type is not SparseDataFrame:
        raise TypeError('Dataset must be a (Pandas) [Sparse]DataFrame')

    sparse_dataset = csr_matrix(dataset.astype(conv_type).to_coo())

    return sparse_dataset


def to_numpy_array(dataset):
    """ Takes a list of Pandas DataFrames or Series' and converts to numpy arrays """
    if type(dataset) is not list:
        dataset = [dataset]

    numpy_matrices = []
    for matrix in dataset:
        if type(matrix) not in [DataFrame, SparseDataFrame, Series]:
            raise TypeError('One of the passed datasets is not a (Pandas) [Sparse]DataFrame or Series')

        numpy_matrices.append(array(matrix.to_list()))

    return numpy_matrices


def prepare_doc_matrix(document_matrix, is_abusive):
    """ Takes a document term matrix, normalizes the rows, and converts it to a CSR matrix """
    # Split dataset into training and testing portions
    (train_matrix, test_matrix), (train_label, test_label) \
        = split_sets(document_matrix, lambda docs: docs, labels=is_abusive)

    # Convert dataset to sparse matrices
    sparse_train, sparse_test = to_csr_matrix(train_matrix), to_csr_matrix(test_matrix)
    normalize_doc_term([sparse_train, sparse_test])

    return (sparse_train, train_label), (sparse_test, test_label)


def load_xgboost_model(filename):
    """ Loads and initializes an XGBoost model """
    model = XGBClassifier(objective='binary:logistic', n_estimators=600, silent=True)
    model.load_model(str(filename))
    model._le = LabelEncoder().fit([0, 1])

    return model


def match_feature_weights(features, weights):
    """ Generates a matching list of features and their weights in a given model """
    weights = [(int(key[1:]), weights[key]) for key in weights]
    feature_weights = sorted(
        [(features[ind], weight) for (ind, weight) in weights],
        key=lambda feat: feat[1],
        reverse=True
    )

    return feature_weights
