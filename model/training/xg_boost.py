from xgboost import XGBClassifier
from pandas import SparseDataFrame
from utilities.data_management import prepare_doc_matrix


# TODO play with number of estimators (and other) within the XGBoost parameters
def train_xg_boost(document_matrix, is_abusive, return_data=False, prepared=False, quiet=True):
    if type(document_matrix) is not SparseDataFrame and not prepared:
        raise TypeError('Document matrix must be a (Pandas) SparseDataFrame.')

    if not prepared:
        (sparse_train, train_label), (sparse_test, test_label) = prepare_doc_matrix(document_matrix, is_abusive)
    else:
        sparse_train, train_label = document_matrix, is_abusive
        sparse_test, test_label = None, None

    # Initialize and train XGBoost model
    classifier = XGBClassifier(objective='binary:logistic', n_estimators=600, silent=quiet)
    classifier.fit(sparse_train, train_label, verbose=not quiet)

    if return_data:
        return classifier, ((sparse_train, train_label), (sparse_test, test_label))
    return classifier
