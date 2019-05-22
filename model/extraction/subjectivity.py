from pandas import DataFrame, SparseDataFrame
from sklearn.feature_extraction.text import CountVectorizer
from numpy import nan


def cond_mult(val):
    """ If the value is not NaN the value is set to 2 """
    return nan if val is nan else val * 2


def subjectivity(dataset, lexicon):
    """ Constructs a document-term matrix using the subjectivity lexicon """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) Dataframe')

    dictionary = lexicon['word']
    values = lexicon['score']

    vectorizer = CountVectorizer(vocabulary=dictionary)
    vector_data = vectorizer.transform(dataset['document_content'])

    document_matrix = SparseDataFrame(vector_data, columns=vectorizer.get_feature_names())

    for ind, col in enumerate(document_matrix.columns):
        if values[ind] != 1:
            document_matrix[col] = document_matrix[col].apply(cond_mult)

    return document_matrix
