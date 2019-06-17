from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer


def hurtlex(dataset, lexicon):
    """ Constructs a document-term matrix using the Hurtlex dictionary """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) Dataframe')

    # TODO consider lemming or stemming the source text to better match the dictionary
    dictionary = lexicon['word']

    vectorizer = CountVectorizer(vocabulary=dictionary)
    vector_data = vectorizer.transform(dataset['document_content'])

    return vector_data
