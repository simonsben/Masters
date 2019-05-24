from xgboost import XGBClassifier
from pandas import SparseDataFrame
from utilities.data_management import split_sets, to_csr_matrix, normalize_doc_term


# TODO play with number of estimators (and other) within the XGBoost parameters
def train_xg_boost(document_matrix, is_abusive):
    if type(document_matrix) is not SparseDataFrame:
        raise TypeError('Document matrix must be a (Pandas) SparseDataFrame.')

    # Split dataset into training and testing portions
    (train_matrix, test_matrix), (train_label, test_label) \
        = split_sets(document_matrix, lambda docs: docs, labels=is_abusive)

    # Convert dataset to sparse matrices
    sparse_train, sparse_test = to_csr_matrix(train_matrix), to_csr_matrix(test_matrix)
    normalize_doc_term([sparse_train, sparse_test])

    # Initialize and train XGBoost model
    classifier = XGBClassifier(objective='binary:logistic', n_estimators=600, silent=True)
    classifier.fit(sparse_train, train_label,
                   eval_set=[(sparse_train, train_label), (sparse_test, test_label)], verbose=False)

    return classifier
