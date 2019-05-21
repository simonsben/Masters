from spacy import load
from pandas import DataFrame, SparseDataFrame
from utilities.data_management import get_content
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


# def process(content):
#     tokens = processor(content)
#
#     num_propositions = sum(1 for token in tokens if token.pos_ == 'PRON')
#     if num_propositions < 2:
#         return None
#
#     return [token for token in tokens if token.pos_ in othering_pos]

def gen_dep(token):
    return '-'.join([token.text, str(token.dep_), str(token.head)]).lower()


def filter_tokens(tokens):
    num_propositions = sum(1 for token in tokens if token.pos_ == 'PRON')
    if num_propositions < 2:
        return []

    terms = []
    for token in tokens:
        if token.pos_ in othering_pos:
            terms.append(token.text)
        if token.dep_ in othering_dep:
            terms.append(gen_dep(token))

    return terms


def othering_dictionary(tokenized, max_terms=10000):
    dictionary = {}
    for tokens in tokenized:
        f_tokens = filter_tokens(tokens)
        for token in f_tokens:
            if token in dictionary:
                dictionary[token] += 1
            else:
                dictionary[token] = 1

    dictionary = sorted(dictionary.items(), key=lambda term: term[1], reverse=True)
    return [term for term, _ in dictionary[:max_terms]]


def othering_vector(dataset):
    """ Takes dataset and tags it with othering features """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) DataFrame')

    # Initialize SpaCy processor and tag documents
    processor = load('en_core_web_sm')
    tokenized = DataFrame(
        get_content(dataset).applymap(processor)
    )

    # Generate othering dictionary
    dictionary = othering_dictionary(tokenized['document_content'])
    print('dictionary', dictionary)

    vectorizer = CountVectorizer(vocabulary=dictionary)
    vector_data = vectorizer.transform(
        dataset['document_content'].astype('|S')
    )

    document_matrix = SparseDataFrame(vector_data, columns=vectorizer.get_feature_names())
    print('document matrix', document_matrix)
