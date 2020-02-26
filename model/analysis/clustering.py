from sklearn.decomposition import PCA
from model.analysis import cluster_verbs


def reduce_and_cluster(vectors, mask=None, num_verbs=50, num_dimensions=100):
    pca = PCA(random_state=420)
    reduced_vectors = pca.fit_transform(vectors)

    if mask is not None:
        reduced_vectors = reduced_vectors[mask]

    cluster_model = cluster_verbs(reduced_vectors, num_verbs, num_dimensions)

    return cluster_model, reduced_vectors
