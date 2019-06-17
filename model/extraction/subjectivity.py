from pandas import DataFrame, SparseDataFrame
from sklearn.feature_extraction.text import CountVectorizer


def subjectivity(dataset, lexicon):
    """ Constructs a document-term matrix using the subjectivity lexicon """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) Dataframe')

    dictionary = lexicon['word']
    scores = lexicon['score']

    vectorizer = CountVectorizer(vocabulary=dictionary)
    document_matrix = vectorizer.transform(dataset['document_content'])

    cols = [ind for ind, val in enumerate(scores) if val != 1]
    document_matrix[:, cols] *= 2

    return document_matrix
