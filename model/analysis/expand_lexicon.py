from utilities.analysis import svd_embeddings, get_nearest_neighbours
from scipy.cluster.vq import whiten, kmeans
from scipy.spatial.distance import euclidean
from scipy.linalg import norm
from numpy import percentile, logical_not

x_key, y_key = 'euclidean_distances', 'cosine_distances'


def cluster_neighbours(neighbours):
    """ Calculate the cluster of embeddings around a given word """
    # Normalize data
    neighbours = neighbours.iloc[1:]
    normed_data = whiten(neighbours[[x_key, y_key]].values)

    # Threshold data
    e_distances = normed_data[:, 0]
    threshold = percentile(e_distances, 20)
    normed_data = normed_data[e_distances < threshold]

    # Cluster
    centroids, distortion = kmeans(normed_data, 2)

    # Split data by centroid
    (a, b) = centroids
    lengths = [(euclidean(point, a), euclidean(point, b)) for point in normed_data]
    is_set_one = [length[0] < length[1] for length in lengths]

    close_set = is_set_one if norm(a) < norm(b) else logical_not(is_set_one)
    print('Adding', neighbours['words'].iloc[1:].loc[close_set].values)

    return list(neighbours['words'].loc[close_set].values)


def expand_lexicon(lexicon, embeddings, simple_expand=None):
    """
    Expand lexicon using trained word embeddings
    :param lexicon: List of word embeddings
    :param embeddings: Pandas DataFrame of words and their embeddings
    :param simple_expand: Specify number of terms to take 'blindly' from around the target word, (default, don't use)
    :return: List of words in the expanded lexicon
    """
    # Normalize embeddings and expand lexicon
    embeddings = svd_embeddings(embeddings)
    if not simple_expand:
        expanded_lexicon = [
            cluster_neighbours(
                get_nearest_neighbours(embeddings, word)[0]
            )
            for word in lexicon
        ]
    else:
        expanded_lexicon = [
            get_nearest_neighbours(embeddings, word, n_words=simple_expand)[0]
            for word in lexicon
        ]

    return expanded_lexicon
