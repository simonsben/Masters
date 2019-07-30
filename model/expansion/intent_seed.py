from spacy import load
from numpy import zeros, asarray, squeeze, logical_not, add, percentile, sum
from itertools import compress
from multiprocessing import Pool
from model.extraction import generate_content_matrix

intent_lead_terms = {'going', 'want', 'need', 'love', 'try',  'tempted', 'like', 'have', 'wish', 'got', 'hope',
                     'hoping', 'trying', 'gon', 'intend', 'wanted', 'tried', 'decided', 'ought', 'meaning'}


# TODO clean up function
def identify_basic_intent(parsed):
    hits = []
    for ind, token in enumerate(parsed):
        if token.tag_ == 'TO' and token.head.pos_ == 'VERB' and token.head.head.pos_ == 'VERB':
            hits.append((token.head.head.text, token.text, token.head.text))
    return hits


def worker_init(*props):
    global parser
    parser = load('en_core_web_md')


def tag_document(props):
    index, context, lead_terms = props

    # Parse context
    parsed = parser(context)

    # Check for basic intent structure
    target = identify_basic_intent(parsed)

    # For basic structure in context
    for hit in target:
        # If leading variable is in the basic intent terms set
        if hit[0] in lead_terms:
            return index
    return None


def break_and_tag(contexts, lead_terms=intent_lead_terms):
    from utilities.data_management import load_execution_params

    # For document in corpus
    worker_pool = Pool(load_execution_params()['n_threads'], initializer=worker_init)
    intent_indexes = worker_pool.map(
        tag_document,
        ((index, context, lead_terms) for index, context in enumerate(contexts))
    )
    worker_pool.close()
    worker_pool.join()

    intent_indexes = filter(lambda index: index is not None, intent_indexes)

    # Convert intention index list to boolean array
    intent_mask = zeros(len(contexts), dtype=bool)
    intent_mask[list(intent_indexes)] = True

    print('intent percentage', sum(intent_mask) / len(contexts))

    return intent_mask


def get_intent_terms(contexts, intent_mask=None, content_data=None):
    if intent_mask is None:
        # Get contexts with intent
        intent_mask = break_and_tag(contexts)
        print('Mask computed, running doc matrix')

    document_matrix, features = generate_content_matrix(contexts) if content_data is None else content_data

    # Get mask for contexts without intent
    no_intent_mask = logical_not(intent_mask)

    # Get term sums for intent and unlabelled documents
    labeled_freq = squeeze(asarray(document_matrix[intent_mask].sum(axis=0)))
    unlabelled_freq = squeeze(asarray(document_matrix[no_intent_mask].sum(axis=0)))

    # Generate mask for the 10th percentile terms
    term_sums = add(labeled_freq, unlabelled_freq)
    threshold = percentile(term_sums, 10)
    frequency_mask = term_sums > threshold

    # Get number of labelled and unlabelled documents
    num_labelled = labeled_freq.shape[0]
    num_unlabelled = unlabelled_freq.shape[0]

    # Assemble significant terms
    relevant_contexts = compress(zip(features, labeled_freq, unlabelled_freq), frequency_mask)
    significant_terms = []
    for term, lab_count, unlab_count in relevant_contexts:
        if unlab_count == 0:
            unlab_count = 1

        freq = (lab_count / num_labelled) / (unlab_count / num_unlabelled)

        if freq > 1:
            significant_terms.append((term, freq))

    significant_terms = sorted(significant_terms, key=lambda term: term[1], reverse=True)

    return significant_terms, intent_mask
