from numpy import array
from time import time
from sparse import COO
from pandas import DataFrame



def pull_tokens(documents):
    tokens = {}

    for document in documents:
        for word in document.split(' '):
            token_value = tokens[word] if word in tokens else 0
            tokens[word] = token_value + 1

    return tokens


def combine_tokens(token_sets, max_terms):
    base = token_sets[0]

    for token_set in token_sets:
        for token in token_set:
            token_value = base[token] if token in base else 0
            base[token] = token_value + token_set[token]

    base = sorted([(token, base[token]) for token in base], key=lambda docs: docs[1], reverse=True)
    base = array(base)[:max_terms, 0]
    base = {token: ind for ind, token in enumerate(base)}

    return base


def get_term_vector(document, vocabulary):
    vector = {}
    for term in document.split(' '):
        ind = vocabulary.get(term)
        if ind is None:
            continue
        elif ind in vector:
            vector[ind] += 1
        else:
            vector[ind] = 1

    return vector


def generate_doc_matrix(documents):
    content = documents['document_content'].astype(str)
    start = time()

    p_voc = content.map_partitions(pull_tokens)
    voc = combine_tokens(p_voc.compute(), 10000)

    print(len(voc), 'in', time() - start)

    start = time()

    p_mat = content \
        .map_partitions(lambda documents: documents.apply(get_term_vector, vocabulary=voc)) \
        .map_partitions(list) \
        .map_partitions(DataFrame)

    mat = p_mat.values \
        .map_blocks(COO) \
        .persist()
