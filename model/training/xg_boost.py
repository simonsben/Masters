from xgboost import XGBClassifier
from pandas import SparseDataFrame
from utilities.data_management import split_sets, to_csr_matrix


def train_xg_boost(document_matrix, is_abusive):
    if type(document_matrix) is not SparseDataFrame:
        raise TypeError('Document matrix must be a (Pandas) SparseDataFrame.')

    (train_matrix, test_matrix), (train_label, test_label) = split_sets(document_matrix, lambda docs: docs, labels=is_abusive)
    sparse_train, sparse_test = to_csr_matrix(train_matrix), to_csr_matrix(test_matrix)

    print('Data converted to CSR')

    classifier = XGBClassifier(objective='binary:logistic', n_estimators=600)
    classifier.fit(sparse_train, train_label.to_numpy(),
                   eval_set=[(sparse_train, train_label), (sparse_test, test_label)])

    return classifier
