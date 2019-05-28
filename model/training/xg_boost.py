from xgboost import XGBClassifier
from pandas import SparseDataFrame
from utilities.data_management import prepare_doc_matrix


# TODO play with number of estimators (and other) within the XGBoost parameters
def train_xg_boost(document_matrix, is_abusive, return_data=False):
    if type(document_matrix) is not SparseDataFrame:
        raise TypeError('Document matrix must be a (Pandas) SparseDataFrame.')

    (sparse_train, train_label), (sparse_test, test_label) = prepare_doc_matrix(document_matrix, is_abusive)

    # Initialize and train XGBoost model
    classifier = XGBClassifier(objective='binary:logistic', n_estimators=600, silent=True)
    classifier.fit(sparse_train, train_label, verbose=False)

    if return_data:
        return classifier, ((sparse_train, train_label), (sparse_test, test_label))
    return classifier
