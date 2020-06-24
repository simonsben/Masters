from utilities.pre_processing import split_pattern, clean_acronym, pre_intent_clean
from numpy import asarray

# The shortest form of intent is 'I will X', which contains three terms
min_terms_for_intent = 3


def split_document(document):
    """ Splits documents into sub-components (called contexts) """
    if not isinstance(document, str):
        return []

    # Split document into contexts using regex pattern and apply clean
    contexts = [pre_intent_clean(context) for context in split_pattern.split(clean_acronym(document))]

    # If context contains less than three words add it to the preceding context
    index = 0
    while index < len(contexts):
        # If not the first context and it contains less than the min number of terms
        if index > 0 and len(contexts[index].split(' ')) < min_terms_for_intent:
            # Add context to previous and delete it (thereby also *incrementing* the index)
            contexts[index - 1] += contexts[index]
            del contexts[index]
        # If the context contains enough terms, move to the next one
        else:
            index += 1

    # If document is a single context with less than the min number of terms, discard it
    if len(contexts) == 1 and len(contexts[0].split(' ')) < min_terms_for_intent:
        return []
    return contexts


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
    for document_index, document in enumerate(documents):
        # Split document into non-zero length contexts
        document_contexts = list(filter(
            lambda content: len(content) > 0 or content == ' ', split_document(document)
        ))

        # Compute index of contexts and get index of the original document
        corpus_index = document_index if original_indexes is None else original_indexes[document_index]

        # Add context mapping to dictionary
        for context_index, _ in enumerate(document_contexts):
            document_indexes.append((corpus_index, context_index))

        # Add the contexts to the list
        corpus_contexts += document_contexts

    document_indexes = asarray(document_indexes).transpose()
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
