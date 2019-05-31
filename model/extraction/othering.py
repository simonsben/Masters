from spacy import load
from pandas import DataFrame, SparseDataFrame
from sklearn.feature_extraction.text import CountVectorizer

othering_pos = {
    'NOUN',
    'PROPN',
    'ADV',
    'ADJ',
    'VERB'
}
othering_dep = {
    'nsubj',
    'nsubjpass',
    'dobj',
    'nmod',
    'npmod',
    'advmod',
    'det',
    'compound'
}

adverb_pos = {
    'ADV'
}
adverb_dep = {
    'advmod'
}


def gen_dep(token):
    """ Convert a relation to a string of the form child-relation-parent """
    return '-'.join([token.text, str(token.dep_), str(token.head)]).lower()


def filter_tokens(tokens, pos=othering_pos, dep=othering_dep):
    """ Filters tokens and returns a string with remaining terms """
    terms = []
    for token in tokens:
        if token.pos_ in othering_pos:
            terms.append(token.text)
        if token.dep_ in othering_dep:
            terms.append(gen_dep(token))

    return ' '.join(terms)


def count_pronouns(tokens):
    """ Returns a bool indicating whether the document has at least two pronouns """
    num_propositions = sum(1 for token in tokens if token.pos_ == 'PRON')
    return num_propositions >= 2


def othering_vectorizer(tokenized, max_terms=10000, token_filter=filter_tokens):
    """ Generates the othering vectorizer from the filtered document tokens """
    vectorizer = CountVectorizer(max_features=max_terms, token_pattern=r'\b\w{2,}[a-zA-Z\-]+\b',
                                 lowercase=False)

    tokenized['multi_props'] = tokenized['document_content'].apply(count_pronouns)
    tokenized['split_content'] = tokenized['document_content'].apply(token_filter)

    vectorizer.fit(tokenized['split_content'] * tokenized['multi_props'])

    return vectorizer


def othering_matrix(dataset, token_filter=None):
    """ Takes dataset and tags it with othering features """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) DataFrame')

    # Initialize SpaCy processor and tag documents
    processor = load('en_core_web_sm')
    tokenized = DataFrame(
        dataset['document_content'].apply(lambda row: processor(row))
    )

    if token_filter is not None:
        vectorizer = othering_vectorizer(tokenized, token_filter=token_filter)
    else:
        vectorizer = othering_vectorizer(tokenized)

    vector_data = vectorizer.transform(
        tokenized['split_content']
    )

    document_matrix = SparseDataFrame(vector_data, columns=vectorizer.get_feature_names())
    return document_matrix


def adverb_matrix(dataset):
    adverb_filter = lambda tokens: filter_tokens(tokens, adverb_pos, adverb_dep)
    return othering_matrix(dataset, adverb_filter)
