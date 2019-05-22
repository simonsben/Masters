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
    'nounmod',
    'npmod',
    'advmod',
    'det',
    'compound'
}


def gen_dep(token):
    """ Convert a relation to a string of the form child-relation-parent """
    return '-'.join([token.text, str(token.dep_), str(token.head)]).lower()


def filter_tokens(tokens):
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


def othering_vectorizer(tokenized, max_terms=10000):
    """ Generates the othering vectorizer from the filtered document tokens """
    vectorizer = CountVectorizer(max_features=max_terms, token_pattern=r'\b\w{2,}[a-zA-Z\-]+\b',
                                 lowercase=False)

    tokenized['multi_props'] = tokenized['document_content'].apply(count_pronouns)
    tokenized['split_content'] = tokenized['document_content'].apply(filter_tokens)

    vectorizer.fit(tokenized['split_content'] * tokenized['multi_props'])

    return vectorizer


def othering_vector(dataset):
    """ Takes dataset and tags it with othering features """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) DataFrame')

    # Initialize SpaCy processor and tag documents
    processor = load('en_core_web_sm')
    tokenized = DataFrame(
        dataset['document_content'].apply(lambda row: processor(row))
    )
    vectorizer = othering_vectorizer(tokenized)

    vector_data = vectorizer.transform(
        tokenized['split_content']
    )

    document_matrix = SparseDataFrame(vector_data, columns=vectorizer.get_feature_names())
    return document_matrix
