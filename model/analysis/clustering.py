from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


def reduce_and_cluster(vectors, mask=None, max_verbs=50, num_dimensions=None):
    pca = PCA(random_state=420)
    reduced_vectors = pca.fit_transform(vectors)

    if mask is not None:
        reduced_vectors = reduced_vectors[mask]

    cluster_model = cluster_verbs(reduced_vectors, max_verbs, num_dimensions)

    return cluster_model, reduced_vectors


def cluster_verbs(verb_vectors, max_verbs=50, num_dimensions=None):
    """ Clusters verb vectors using hierarchical clustering """
    model = AgglomerativeClustering(distance_threshold=1, n_clusters=None, affinity='cosine', linkage='complete')

    if num_dimensions is not None:
        verb_vectors = verb_vectors[:, :num_dimensions]
    if max_verbs is not None:
        verb_vectors = verb_vectors[:max_verbs]

    model.fit(verb_vectors)

    return model
