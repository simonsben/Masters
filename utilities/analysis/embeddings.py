from scipy.spatial.distance import cosine, euclidean
from scipy.linalg import norm


def normalize_embeddings(vectors):
    """ Normalizes the embeddings (converts to z-scores) """
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    vectors = (vectors - mean) / std

    return vectors


def calculate_cosine_distance(vectors, target):
    def calc_cosine(vector): return cosine(vector, target)
    return vectors.map_partitions(lambda df: df.apply(calc_cosine, axis=1), meta=float)


def calculate_euclidean_distance(vectors, target):
    def calc_dist(vector): return euclidean(vector, target)
    return vectors.map_partitions(lambda df: df.apply(calc_dist, axis=1), meta=float)


def calculate_norms(vectors):
    return vectors.map_partitions(lambda df: df.apply(norm, axis=1), meta=float)


def threshold_data(datasets, condition_data, threshold):
    selection = condition_data < threshold
    return [dataset[selection] for dataset in datasets]


def get_nearest_neighbours(embeddings, target_word, n_words=50, max_angle=.85, reverse=False, silent=True):
    """
    Calculates the nearest neighbours of a target vector within a vector space of word embeddings
    :param embeddings: Dask dataframe of the word embeddings
    :param target_word: Target word
    :param n_words: Number of words to return
    :param max_angle: Maximum cosine angle
    :param reverse:
    :param silent:
    :return:
    """

    # Define slices
    words = embeddings[['0']]
    vectors = normalize_embeddings(embeddings.iloc[:, 1:])

    target = vectors[words['0'] == target_word].compute()
    if not silent: print('Vectors normalized, filtering')

    # Calculate cosine distances and reduce dataset
    words['cosine_distances'] = calculate_cosine_distance(vectors, target)
    threshold = (2 - max_angle) if reverse else max_angle
    (vectors, words) = threshold_data((vectors, words), words['cosine_distances'], threshold)

    # Calculate euclidean distances
    words['euclidean_distances'] = calculate_euclidean_distance(vectors, target)
    words['vector_norms'] = calculate_norms(vectors)

    operation = words.nlargest if reverse else words.nsmallest
    return operation(n=n_words, columns='euclidean_distances').compute(), target
