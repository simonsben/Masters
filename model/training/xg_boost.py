from xgboost import XGBClassifier
from utilities.data_management import prepare_doc_matrix, load_execution_params
from scipy.sparse import csr_matrix


# TODO run optimization over hyper-parameters
def train_xg_boost(document_matrix, is_abusive, return_data=False, prepared=False, verb=0):
    if not prepared and type(document_matrix) is not csr_matrix:
        raise TypeError('Document matrix must be a CSR matrix.')

    # If not pre-split, split dataset into training and test data
    if not prepared:
        (sparse_train, sparse_test), (train_label, test_label) = prepare_doc_matrix(document_matrix, is_abusive)
    else:
        sparse_train, train_label = document_matrix, is_abusive
        sparse_test, test_label = None, None

    # Initialize and train XGBoost model
    n_threads = load_execution_params()['n_threads']
    classifier = XGBClassifier(objective='binary:logistic', n_estimators=600, verbosity=verb, n_jobs=n_threads)
    classifier.fit(sparse_train, train_label, verbose=(verb > 0))

    if return_data:
        return classifier, ((sparse_train, train_label), (sparse_test, test_label))
    return classifier
