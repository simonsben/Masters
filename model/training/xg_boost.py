from xgboost import XGBClassifier
from pandas import SparseDataFrame
from utilities.data_management import prepare_doc_matrix, load_execution_params
from time import time
from scipy.sparse import csr_matrix


# TODO play with number of estimators (and other) within the XGBoost parameters
def train_xg_boost(document_matrix, is_abusive, return_data=False, prepared=False, verb=0):
    # if type(document_matrix) is not SparseDataFrame and not prepared:
    #     raise TypeError('Document matrix must be a (Pandas) SparseDataFrame.')

    start = time()
    if not prepared:
        # (sparse_train, train_label), (sparse_test, test_label) = prepare_doc_matrix(document_matrix, is_abusive)
        num_rows = document_matrix.shape[0]
        pivot_index = int(num_rows * (1 - .3))

        sparse_train, sparse_test = document_matrix[:pivot_index], document_matrix[pivot_index:]
        train_label, test_label = is_abusive[:pivot_index], is_abusive[:pivot_index]
    else:
        sparse_train, train_label = document_matrix, is_abusive
        sparse_test, test_label = None, None

    print('prepared in', time() - start)
    print(sparse_train.shape, train_label.shape)

    # Initialize and train XGBoost model
    n_threads = load_execution_params()['n_threads']
    classifier = XGBClassifier(objective='binary:logistic', n_estimators=600, verbosity=verb, n_jobs=n_threads)
    print(classifier)

    classifier.fit(sparse_train, train_label, verbose=verb > 0)

    if return_data:
        return classifier, ((sparse_train, train_label), (sparse_test, test_label))
    return classifier
