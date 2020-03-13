from utilities.pre_processing import split_pattern, clean_acronym, pre_intent_clean
from numpy import asarray


def split_document(document):
    """ Splits documents into sub-components (called contexts) """
    if not isinstance(document, str):
        return []

    contexts = split_pattern.split(clean_acronym(document))
    contexts = [pre_intent_clean(context) for context in contexts if len(context.split(' ')) > 1]

    return list(filter(lambda context: len(context) > 1, contexts))


def split_into_contexts(documents, original_indexes=None):
    """
    Splits documents into contexts (sentences)
    :param documents: Iterable collection of documents
    :param original_indexes: Indexes of the original documents, if not enumeration
    :return: List of contexts, Mapping of corpus index to context slice
    """
    document_indexes = []
    corpus_contexts = []

    # For each document in corpus
    for index, document in enumerate(documents):
        # Split document into non-zero length contexts
        document_contexts = list(filter(
            lambda content: len(content) > 0 or content == ' ', split_document(document)
        ))

        # Compute index of contexts and get index of the original document
        corpus_index = index if original_indexes is None else original_indexes[index]

        # Add context mapping to dictionary
        for context_index in range(len(document_contexts)):
            document_indexes.append((corpus_index, context_index))

        # Add the contexts to the list
        corpus_contexts += document_contexts

    document_indexes = asarray(document_indexes)
    return corpus_contexts, document_indexes


def generate_context_matrix(contexts):
    """ Compute context term matrix for word n-grams """
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize document vectorizer
    vectorizer = CountVectorizer(ngram_range=(3, 6), max_features=25000)

    # Construct document matrix
    document_matrix = vectorizer.fit_transform(contexts)
    features = vectorizer.get_feature_names()

    return document_matrix, features
