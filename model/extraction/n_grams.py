from sklearn.feature_extraction.text import CountVectorizer
from utilities.data_management import make_path, check_existence
from pandas import SparseDataFrame, read_csv


def vectorize_source(source_path, num_features=10000, use_words=True):
    """
    Takes a pre-processed file and constructs a sparse document term matrix.

    :param source_path: path of the pre-processed source file
    :param num_features: cut-off for the maximum number of features (retains most frequent), (default 10,000)
    :param use_words: whether to vectorize based on words or character n-grams, (default True)
    :return: document term matrix, sparse dataframe (pandas)
    """
    # Check for source file
    source_path = make_path(source_path)
    check_existence(source_path)

    # Get document content
    documents = read_csv(source_path)['document_content'].astype('|S')

    # Initialize vectorizer
    if use_words:
        vectorizer = CountVectorizer(max_features=num_features)
    else:
        vectorizer = CountVectorizer(max_features=num_features, ngram_range=(3, 5), analyzer='char_wb')

    # Vectorize corpus and convert to a sparse dataframe
    vector_data = vectorizer.fit_transform(documents)
    document_matrix = SparseDataFrame(vector_data, columns=vectorizer.get_feature_names())

    return document_matrix
