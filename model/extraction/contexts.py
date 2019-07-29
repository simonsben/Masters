from re import compile

context_breaks = compile(r'[.?!]+')


def pull_document_contexts(documents):
    """
    Splits documents into contexts (scentences)
    :param documents: Iterable collection of documents
    :return: List of contexts, Mapping of corpus index to context slice
    """
    context_map = {}
    corpus_contexts = []

    # For each document in corpus
    for index, document in enumerate(documents):
        # Split document into non-zero length contexts
        document_contexts = list(filter(
            lambda cont: len(cont) > 0,
            context_breaks.split(document) if isinstance(document, str) else []
        ))

        # Add set of contexts to the map
        corpus_index = len(corpus_contexts)
        context_map[index] = slice(corpus_index, corpus_index + len(document_contexts))

        # Add the contexts to the list
        corpus_contexts += document_contexts

    return corpus_contexts, context_map


def generate_content_matrix(contexts):
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize document vectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)

    # Construct document matrix
    document_matrix = vectorizer.fit_transform(contexts)
    features = vectorizer.get_feature_names()

    return document_matrix, features
