from sklearn.feature_extraction.text import CountVectorizer
from pandas import SparseDataFrame, DataFrame


def n_gram_matrix(dataset, num_features=10000, use_words=True):
    """
    Takes a pre-processed file and constructs a sparse document term matrix.

    :param dataset: Dataset of document content
    :param num_features: cut-off for the maximum number of features (retains most frequent), (default 10,000)
    :param use_words: whether to vectorize based on words or character n-grams, (default True)
    :return: document term matrix, sparse dataframe (pandas)
    """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) DataFrame')

    # Initialize vectorizer
    if use_words:
        vectorizer = CountVectorizer(max_features=num_features)
    else:
        vectorizer = CountVectorizer(max_features=num_features, ngram_range=(3, 5), analyzer='char_wb')

    # Vectorize corpus and convert to a sparse dataframe
    vector_data = vectorizer.fit_transform(dataset['document_content'])
    document_matrix = SparseDataFrame(vector_data, columns=vectorizer.get_feature_names())

    return document_matrix
