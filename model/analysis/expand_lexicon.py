from utilities.analysis import svd_embeddings, get_nearest_neighbours
from scipy.cluster.vq import whiten, kmeans
from scipy.spatial.distance import euclidean
from scipy.linalg import norm
from numpy import percentile, logical_not
from re import match, compile
from nltk.corpus import wordnet


x_key, y_key = 'euclidean_distances', 'cosine_distances'
wordnet_regex = compile(r'\w+\.')


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


def wordnet_expansion(lexicon, n_words=None):
    if n_words is None:
        n_words = 5

    looker = wordnet.synsets
    expanded_lexicon = [
        [str(synonym.lemmas()[0].name()) for synonym in looker(word)[:n_words]]
        for word in lexicon
    ]

    return expanded_lexicon


def embedding_expansion(lexicon, embeddings, simple_expand=None):
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
        expanded_lexicon = []
        for ind, word in enumerate(lexicon):
            new_terms = get_nearest_neighbours(embeddings, word, n_words=(simple_expand + 1))[0]

            if len(new_terms) < 1:
                continue

            new_terms = new_terms['words'].values[1:]
            expanded_lexicon.append(new_terms)

            print('Adding', new_terms, ' - ', round((ind + 1) / len(lexicon) * 10000) / 100, '% complete')

    return lexicon


def expand_lexicon(lexicon, embeddings=None, simple_expand=None):
    # If no embeddings assume wordnet expansion
    if embeddings is None:
        return wordnet_expansion(lexicon, simple_expand)
    # If embeddings supplied assume embedding expansion
    else:
        return embedding_expansion(lexicon, embeddings, simple_expand)
