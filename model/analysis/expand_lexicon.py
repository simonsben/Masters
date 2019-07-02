from utilities.analysis import svd_embeddings
from scipy.cluster.vq import whiten, kmeans


# TODO complete using clustering technique in Colab notebook

def expand_lexicon(lexicon_embeddings, embeddings):
    embeddings = svd_embeddings(embeddings.append(lexicon_embeddings))

