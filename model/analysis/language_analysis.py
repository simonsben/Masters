from numpy import asarray, where, ndarray
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering

english_label = '__label__en'


def classify_languages(documents, model):
    """ Predicts the language in a set of documents """
    document_languages = [
        model.predict(document)[0][0] for document in documents if isinstance(document, str)
    ]

    return document_languages


def get_english_indexes(documents, model, boolean_mask=False):
    """ Gets the indexes of english documents """
    languages = asarray(classify_languages(documents, model))

    is_english = languages == english_label
    if not boolean_mask:
        [is_english] = where(is_english)

    return is_english


def filter_non_english(documents, model, return_indexes=False):
    """ Returns the english documents with (optionally) the associated mask """
    if isinstance(documents, list):
        documents = asarray(documents)
    elif not isinstance(documents, ndarray):
        raise AttributeError('Expected list or numpy array of documents.')

    is_english = get_english_indexes(documents, model)

    if return_indexes:
        return documents[is_english], is_english
    return documents[is_english]


def generate_word_vectors(words, model):
    """ Generates word vectors for a given list of words using a fastText model """
    word_vectors = [
        [word] + list(model.get_word_vector(word))
        for word in words
    ]

    word_vectors = DataFrame(
        word_vectors,
        columns=(
                ['word'] + [str(index) for index in range(model.get_dimension())]
        )
    )

    return word_vectors


def cluster_verbs(verb_vectors, num_top_verbs=30, num_dimensions=50):
    """ Clusters verb vectors using hierarchical clustering """
    model = AgglomerativeClustering(distance_threshold=1, n_clusters=None, affinity='cosine', linkage='average')
    model.fit(verb_vectors[:num_top_verbs, :num_dimensions])

    return model
