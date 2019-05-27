from utilities.data_management import make_path, check_existence
from pandas import DataFrame
from numpy import zeros


def load_vectors(filename):
    """ Takes the fastText vectors and generates a dictionary for them.
    (Almost) directly copied from https://fasttext.cc/
    """
    path = make_path(filename)
    check_existence(path)

    fin = path.open(mode='r', encoding='utf-8', newline='\n', errors='ignore')
    num_words, vector_dimension = map(int, fin.readline().split())
    vectors = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        vectors[tokens[0]] = list(map(float, tokens[1:]))

    return vectors, (num_words, vector_dimension)


def vectorize_content(content, vectors, vector_dim):
    """ Translates document content to word vectors """
    prepared_content = content.split(' ')
    vector_content = zeros((len(prepared_content), vector_dim))

    for ind, token in enumerate(prepared_content):
        vector = vectors.get(token)
        if vector is not None:
            vector_content[ind] = vector

    return vector_content


def vectorize_data(dataset, vectors, vector_dim=300):
    """ Converts a dataset's content to word vectors """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) DataFrame')

    def vectorizer(doc):
        return vectorize_content(doc, vectors, vector_dim)

    dataset['vectorized_content'] = dataset['document_content'].apply(vectorizer)
