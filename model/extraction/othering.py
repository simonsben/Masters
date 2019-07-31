from spacy import load
from pandas import DataFrame
from time import time
from numpy import where
from multiprocessing import Pool
from itertools import chain

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
    # 'advmod'
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


def generate_parse_index():
    """ Generates a dict with tag-to-index mappings """
    inds = {
        tag: str(ind) for ind, tag in enumerate(list(othering_pos) + list(othering_dep))
    }

    return inds


def count_pronouns(tokens):
    """ Returns a bool indicating whether the document has at least two pronouns """
    num_propositions = sum(1 for token in tokens if token.pos_ == 'PRON')
    return num_propositions >= 2


def processor(frame):
    model = load('en_core_web_md')

    parsed_documents = [model(document) for document in frame]
    # print(parsed_documents)
    return parsed_documents


def othering_vectorizer(tokenized, max_terms=10000, token_filter=filter_tokens):
    from sklearn.feature_extraction.text import CountVectorizer

    """ Generates the othering vectorizer from the filtered document tokens """
    vectorizer = CountVectorizer(max_features=max_terms, token_pattern=r'\b\w{2,}[a-zA-Z\-]+\b',
                                 lowercase=False)

    tokenized['multi_props'] = tokenized['document_content'].apply(count_pronouns)
    tokenized['split_content'] = tokenized['document_content'].apply(token_filter)

    vectorizer.fit(
        where(tokenized['multi_props'].values, tokenized['split_content'].values, '')
    )

    return vectorizer


def parse_documents(documents):
    from utilities.data_management.file_management import load_execution_params

    # Load execution parameters and data
    num_threads = load_execution_params()['n_threads']
    num_frames = num_threads * 10   # Define extra frames (not all docs are the same length)
    num_docs = documents.shape[0]
    frame_size = num_docs / num_frames

    # Initialize processing pool and content
    pool = Pool(num_threads)
    frames = [
        documents['document_content'].iloc[int(ind * frame_size):int((ind + 1) * frame_size)]
        for ind in range(num_frames)
    ]

    # Parse content and close pool
    parsed = pool.map(processor, frames)
    pool.close()
    pool.join()

    # Generate dataframe with content
    parsed = DataFrame({'document_content': list(chain.from_iterable(parsed))})

    return parsed


def othering_matrix(dataset, token_filter=None):
    """ Takes dataset and tags it with othering features """
    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) DataFrame')

    # Initialize SpaCy processor and tag documents
    start = time()
    tokenized = parse_documents(dataset)
    print('Spacy parse complete in', time() - start)

    start = time()
    if token_filter is not None:
        vectorizer = othering_vectorizer(tokenized, token_filter=token_filter)
    else:
        vectorizer = othering_vectorizer(tokenized)

    vector_data = vectorizer.transform(
        tokenized['split_content']
    )
    print('Vectorized in', time() - start)

    return vector_data, vectorizer.get_feature_names()


def adverb_matrix(dataset):
    adverb_filter = lambda tokens: filter_tokens(tokens, adverb_pos, adverb_dep)
    return othering_matrix(dataset, adverb_filter)
