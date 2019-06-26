from scipy.linalg import norm
from scipy.spatial.distance import cosine, euclidean
from numpy import abs


def normalize_embeddings(vectors):
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    vectors = (vectors - mean) / std

    return vectors


def get_nearest_neighbours(embeddings, target_word, n_words=150, max_angle=1, reverse=False, silent=True):
    # Define slices
    words = embeddings[[0]].rename(columns={0: 'words'})
    vectors = normalize_embeddings(embeddings.iloc[:, 1:])

    target = vectors[words['words'] == target_word].compute()
    if not silent: print('Vectors normalized, filtering')

    # Define metrics
    def calc_dist(vector): return euclidean(vector, target)
    if reverse:
        def calc_cosine(vector): return abs(2 - cosine(vector, target))
    else:
        def calc_cosine(vector): return cosine(vector, target)

    # Calculate cosine distances and reduce dataset
    words['cosine_distances'] = vectors.map_partitions(lambda df: df.apply(calc_cosine, axis=1), meta=float)
    vectors = vectors[words['cosine_distances'] < max_angle]

    # Calculate euclidean distances
    words['euclidean_distances'] = vectors.map_partitions(lambda df: df.apply(calc_dist, axis=1), meta=float)

    return words.nsmallest(n=n_words, columns='euclidean_distances').compute(), norm(target)
