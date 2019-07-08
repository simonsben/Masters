from spacy import load
from pandas import DataFrame

content = 'Hello, how it going for you today? Hello, how it going for you today? Hello, how it going for you today?'


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


def generate_parse_index():
    """ Generates a dict with tag-to-index mappings """
    inds = {
        tag: str(ind) for ind, tag in enumerate(list(othering_pos) + list(othering_dep))
    }

    return inds


def filter_to_save(parsed_documents):
    """ Filters tokens and returns a string with remaining terms and tags """
    inds = generate_parse_index()

    filtered_documents = []
    for tokens in parsed_documents:
        terms = []
        for token in tokens:
            if token.pos_ in othering_pos:
                terms.append(token.text + '%' + inds[token.pos_])
            if token.dep_ in othering_dep:
                terms.append(gen_dep(token) + '%' + inds[token.dep_])
        filtered_documents.append(' '.join(terms))

    return filtered_documents


def processor(frame):
    model = load('en_core_web_sm')

    return filter_to_save([model(document) for document in frame])


if __name__ == '__main__':
    from utilities.data_management import open_w_pandas, move_to_root, make_path, check_existence, check_writable, \
        load_execution_params
    from time import time
    from multiprocessing import Pool
    from itertools import chain

    move_to_root()

    # Load params
    params = load_execution_params()
    dataset_name = params['dataset']

    # Define paths
    data_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
    dest_path = make_path('data/processed_data/') / dataset_name / 'derived' / 'spacy_parse.csv'

    # Check paths
    check_existence(data_path)
    check_writable(dest_path)

    # Load model and data
    data = open_w_pandas(data_path)

    num_threads = 38
    num_frames = num_threads * 15
    num_docs = data.shape[0]
    frame_size = num_docs / num_frames
    frames = [
                 data['document_content'].iloc[int(ind * frame_size):int((ind + 1) * frame_size)]
                 for ind in range(num_frames)
             ]
    print('Data and model loaded, starting')

    # Parse
    pool = Pool(num_threads)
    start = time()

    parsed = list(chain.from_iterable(pool.map(processor, frames)))
    pool.close()
    pool.join()
    print('Parsed in', time() - start)

    parsed = DataFrame(parsed, columns=['parsed_content'])
    print(parsed)

    # Save parsed
    parsed.to_csv(dest_path)
