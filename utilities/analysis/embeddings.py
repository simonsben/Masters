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


def calculate_euclidean_distance(vectors, target, weights=None):
    """ Calculates the euclidean distance of each vector from a target vector """
    def calc_dist(vector): return euclidean(vector, target, w=weights)
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
    target = vectors[words == target_word].compute().values

    # If word isn't in the set of embeddings, return None
    if len(target) < 1:
        return None
    return target[0]


def merge_vectors(vectors):
    """ Merge a vector into another one """
    scale = mean([norm(vector) for vector in vectors])
    return sum([vector / norm(vector) for vector in vectors], axis=0) * scale


def svd_embeddings(raw_embeddings, dimensions=50, get_s=False):
    """ Takes word embeddings and converts """
    # Calculate Word-Topic matrix
    vectors, s, _ = svd(
        normalize_embeddings(raw_embeddings.iloc[:, 1:].values)
    )
    vectors = vectors[:, :dimensions].persist()

    # Form new word-vector Dask Dataframe
    embeddings = raw_embeddings[['words']]
    embeddings[[str(ind) for ind in range(1, vectors.shape[1] + 1)]] = from_array(vectors)

    if not get_s:
        return embeddings
    return embeddings, s.compute()[:dimensions]


def get_nearest_neighbours(embeddings, target_word, n_words=250, max_angle=.7, normal=True, weights=None):
    """
    Calculates the nearest neighbours of a target vector within a vector space of word embeddings
    :param embeddings: Dask dataframe of the word embeddings
    :param target_word: Target word
    :param n_words: Number of words to return
    :param max_angle: Maximum cosine angle
    :param normal: Whether the embeddings passed are normalized
    :param weights: Weights for use in the euclidean distance calculation, (default, not used)
    :return: N closest vectors, target vector
    """

    # Define data
    vectors = embeddings.iloc[:, 1:] if normal else normalize_embeddings(embeddings.iloc[:, 1:])

    cols = [str(ind) for ind in range(1, vectors.shape[1] + 1)]
    words = embeddings[['words'] + cols]
    target = get_word_vector(vectors, words['words'], target_word) if type(target_word) is str else target_word

    if target is None:
        return [], None

    # Calculate cosine distances and reduce dataset
    words['cosine_distances'] = calculate_cosine_distance(vectors, target)
    (vectors, words) = threshold_data((vectors, words), words['cosine_distances'], max_angle)

    # Calculate euclidean distances
    words['euclidean_distances'] = calculate_euclidean_distance(vectors, target, weights)
    words['vector_norms'] = calculate_norms(vectors)

    # Select N closest vectors
    neighbours = words.nsmallest(n=n_words, columns='euclidean_distances').compute().reset_index(drop=True)

    return neighbours, target


def get_relative_neighbours(embeddings, ref_words, n_words=50, cos_distance=.5, normal=True):
    """
    Calculate equivalent vector (ex. Toronto is to Ontario what ??? is to England, or
    (England, Toronto, Ontario) -> London)
    :param embeddings: Dask dataframe of term embeddings
    :param ref_words: Reference words (target, desired_reference, target_reference)
    :param n_words: Number of words to return
    :param cos_distance: Maximum cosine distance from desired target
    :param normal: Whether the provided embeddings are normalized (z-scored)
    :return: N nearest neighbours
    """
    # Define data
    words = embeddings[['words']]
    vectors = embeddings.iloc[:, 1:] if normal else normalize_embeddings(embeddings.iloc[:, 1:])

    # Calculate theoretical location of the desired target vector
    (target, ref_target, ref_related) = [get_word_vector(vectors, words['words'], word) for word in ref_words]
    related = merge_vectors((target, -ref_target, ref_related))

    # Calculate nearest neighbours of the desired theoretical location
    embeddings[[str(ind) for ind in range(1, 301)]] = vectors
    return get_nearest_neighbours(embeddings, related, n_words=n_words, max_angle=cos_distance, normal=True)
