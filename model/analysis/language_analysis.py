from numpy import asarray, where, ndarray
from pandas import DataFrame

english_label = '__label__en'


def classify_languages(documents, model):
    """ Predicts the language in a set of documents """
    document_languages = [
        model.predict(document)[0][0] for document in documents if isinstance(document, str)
    ]

    return document_languages


def get_english_indexes(documents, model):
    """ Gets the indexes of english documents """
    languages = asarray(classify_languages(documents, model))
    [is_english] = where(languages == english_label)

    return is_english


def filter_non_english(documents, model, return_indexes=False):
    """ Returns the english documents """
    if isinstance(documents, list):
        documents = asarray(documents)
    elif not isinstance(documents, ndarray):
        raise AttributeError('Expected list or numpy array of documents.')

    is_english = get_english_indexes(documents, model)

    if return_indexes:
        return documents[is_english], is_english
    return documents[is_english]


def generate_word_vectors(words, model):
    word_vectors = [
        [word] + list(model.get_word_vector(word))
        for word in words if isinstance(word, str)
    ]

    word_vectors = DataFrame(
        word_vectors,
        columns=(
                ['word'] + [str(index) for index in range(model.get_dimension())]
        )
    )

    return word_vectors
