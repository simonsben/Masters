from re import compile

context_pattern = compile(r"(?:[\w']+[;:,.?!]\s+)?(?:[\w'-]+\b[\s-]*){2,}")
repeats = compile(r'(.)(\1{2,})')
cleaner = compile(r'[^a-zA-Z ]')
acronym = compile(r'(\w\.){2,}')


def clean_acronym(document):
    """ Removes periods from acronyms (ex. U.S.A.) """
    return acronym.sub(lambda match: match[0].replace('.', ''), document)


def split_document(document):
    """ Splits document using a more aggressive definition """
    if not isinstance(document, str):
        return []

    tmp = clean_acronym(repeats.sub('', document))

    matches, start = [], 0
    match = context_pattern.search(tmp, start)

    while match is not None:
        matches.append(match[0])
        start = match.end()
        match = context_pattern.search(tmp, start)

    matches = [cleaner.sub('', doc) for doc in matches]
    return matches


def pull_document_contexts(documents):
    """
    Splits documents into contexts (sentences)
    :param documents: Iterable collection of documents
    :return: List of contexts, Mapping of corpus index to context slice
    """
    context_map = {}
    corpus_contexts = []

    # For each document in corpus
    for index, document in enumerate(documents):
        # Split document into non-zero length contexts
        document_contexts = list(filter(
            lambda cont: len(cont) > 0, split_document(document)
        ))

        # Add set of contexts to the map
        corpus_index = len(corpus_contexts)
        context_map[index] = slice(corpus_index, corpus_index + len(document_contexts))

        # Add the contexts to the list
        corpus_contexts += document_contexts

    return corpus_contexts, context_map


def generate_context_matrix(contexts):
    """ Compute context term matrix for word unigrams and bigrams """
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize document vectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=25000)

    # Construct document matrix
    document_matrix = vectorizer.fit_transform(contexts)
    features = vectorizer.get_feature_names()

    return document_matrix, features
