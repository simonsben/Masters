from scipy.spatial.distance import cosine, euclidean
from scipy.linalg import norm
from numpy import sum, mean
from dask.dataframe import from_array
from dask.array.linalg import svd


def normalize_embeddings(vectors):
    """ Normalizes the embeddings (converts to z-scores) """
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)

    return (vectors - mean) / std


def calculate_cosine_distance(vectors, target):
    """ Calculate the cosine distance of each vector from a target vector """
    def calc_cosine(vector): return cosine(vector, target)
    return vectors.map_partitions(lambda df: df.apply(calc_cosine, axis=1), meta=float)


def calculate_euclidean_distance(vectors, target):
    """ Calculates the euclidean distance of each vector from a target vector """
    def calc_dist(vector): return euclidean(vector, target)
    return vectors.map_partitions(lambda df: df.apply(calc_dist, axis=1), meta=float)


def calculate_norms(vectors):
    """ Calculates the euclidean norms of each vectors """
    return vectors.map_partitions(lambda df: df.apply(norm, axis=1), meta=float)


def threshold_data(datasets, condition_data, threshold):
    """ Threshold a set of data """
    selection = condition_data < threshold
    return [dataset[selection] for dataset in datasets]


def get_word_vector(vectors, words, target_word):
    """ Compute the word embedding for a given target word """
    return vectors[words == target_word].compute().values[0]


def merge_vectors(vectors):
    """ Merge a vector into another one """
    scale = mean([norm(vector) for vector in vectors])
    return sum([vector / norm(vector) for vector in vectors], axis=0) * scale


def embeddings_to_svd(raw_embeddings, dimensions=50):
    """ Takes word embeddings and converts """
    vectors, _, _ = svd(
        normalize_embeddings(raw_embeddings.iloc[:, 1:].values)
    )
    vectors = vectors.persist()[:, :dimensions]

    embeddings = raw_embeddings[['words']]
    embeddings[[str(ind) for ind in range(1, vectors.shape[1] + 1)]] = from_array(vectors)
    return embeddings


def get_nearest_neighbours(embeddings, target_word, n_words=50, max_angle=.7, reverse=False, normal=True):
    """
    Calculates the nearest neighbours of a target vector within a vector space of word embeddings
    :param embeddings: Dask dataframe of the word embeddings
    :param target_word: Target word
    :param n_words: Number of words to return
    :param max_angle: Maximum cosine angle
    :param reverse: Whether to get the target word, or its inverse
    :param normal: Whether the embeddings passed are normalized
    :return: N cloasest vectors, target vector
    """
    # Define data
    words = embeddings[['words']]
    vectors = embeddings.iloc[:, 1:] if normal else normalize_embeddings(embeddings.iloc[:, 1:])
    target = get_word_vector(vectors, words['words'], target_word) if type(target_word) is str else target_word
    if reverse: target *= -1

    # Calculate cosine distances and reduce dataset
    words['cosine_distances'] = calculate_cosine_distance(vectors, target)
    # threshold = (2 - max_angle) if reverse else max_angle
    threshold = max_angle
    (vectors, words) = threshold_data((vectors, words), words['cosine_distances'], threshold)

    # Calculate euclidean distances
    words['euclidean_distances'] = calculate_euclidean_distance(vectors, target)
    words['vector_norms'] = calculate_norms(vectors)

    # operation = words.nlargest if reverse else words.nsmallest
    operation = words.nsmallest
    neighbours = operation(n=n_words, columns='euclidean_distances').compute().reset_index(drop=True)
    return neighbours, target


def get_relative_neighbours(embeddings, ref_words, n_words=50, max_angle=.85, normal=True):
    # Define slices
    words = embeddings[['words']]
    vectors = embeddings.iloc[:, 1:] if normal else normalize_embeddings(embeddings.iloc[:, 1:])

    (target, ref_target, ref_related) = [get_word_vector(vectors, words['words'], word) for word in ref_words]
    related = merge_vectors((target, -ref_target, ref_related))

    embeddings[[str(ind) for ind in range(1, 301)]] = vectors
    return get_nearest_neighbours(embeddings, related, n_words=n_words, max_angle=max_angle, normal=True)
