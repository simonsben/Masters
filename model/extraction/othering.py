from spacy import load
from pandas import DataFrame
from time import time
from multiprocessing import Pool
from utilities.data_management.file_management import load_execution_params
from sklearn.feature_extraction.text import CountVectorizer
from itertools import compress

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
        if token.pos_ in pos:
            terms.append(token.text)
        if token.dep_ in dep:
            terms.append(gen_dep(token))

    return ' '.join(terms)


def contains_pronouns(tokens):
    """ Returns a bool indicating whether the document has at least two pronouns """
    num_propositions = sum(1 for token in tokens if token.pos_ == 'PRON')
    return num_propositions >= 2


def worker_process(package):
    document, document_filter = package
    parsed = parser(document)

    has_pronouns = contains_pronouns(parsed)
    filtered = document_filter(parsed)

    return filtered, has_pronouns


def init_workers():
    global parser
    parser = load('en_core_web_md')


def parse_documents(documents, document_filter):
    # Initialize processing pool and content
    workers = Pool(load_execution_params()['n_threads'], initializer=init_workers)
    document_filter = document_filter if document_filter is not None else filter_tokens

    # Parse content and close pool
    parsed = list(
        workers.imap(
            worker_process,
            ((document, document_filter) for document in documents['document_content'].values),
            chunksize=250
        )
    )
    workers.close()
    workers.join()
    print('Pool done')

    documents, has_pronouns = map(list, zip(*parsed))

    # Generate DataFrame with content
    parsed = DataFrame({'multi_props': has_pronouns, 'split_content': documents})

    return parsed


def othering_vectorizer(tokenized, max_terms=10000):
    """ Generates the othering vectorizer from the filtered document tokens """
    vectorizer = CountVectorizer(max_features=max_terms, token_pattern=r'\b\w{2,}[a-zA-Z\-]+\b',
                                 lowercase=False)
    vectorizer.fit(
        list(compress(tokenized['split_content'].values, tokenized['multi_props'].values))
    )

    return vectorizer


def othering_matrix(dataset, token_filter=None):
    """ Takes dataset and tags it with othering features """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) DataFrame')

    # Initialize SpaCy processor and tag documents
    start = time()
    tokenized = parse_documents(dataset, token_filter)
    print('Spacy parse complete in', time() - start)

    start = time()
    vectorizer = othering_vectorizer(tokenized)
    vector_data = vectorizer.transform(
        tokenized['split_content']
    )
    print('Vectorized in', time() - start)

    return vector_data, vectorizer.get_feature_names()


def adverb_filter(tokens):
    return filter_tokens(tokens, adverb_pos, adverb_dep)


def adverb_matrix(dataset):
    return othering_matrix(dataset, adverb_filter)
