from spacy import load
from numpy import zeros, asarray, squeeze, logical_not, add, percentile, sum
from itertools import compress
from multiprocessing import Pool
from model.extraction import generate_context_matrix

first_person = {'i', 'we', 'me', 'us', 'em', 'mine', 'myself'}


def identify_basic_intent(parsed):
    """ Determines if parsed document contains a sequence of term that indicate intent """
    # For each token in parsed document
    for ind, token in enumerate(parsed):
        # If basic intent form of VERB .. TO .. VERB (where tokens are related)
        if token.tag_ == 'TO' and token.head.pos_ == 'VERB' and token.head.head.pos_ == 'VERB':
            is_first = False

            # For each child token of first VERB term
            for child in token.head.head.children:

                # If first verb is related to a first person pronoun
                if child.dep_ == 'nsubj' and child.text in first_person:
                    is_first = True
                elif child.dep_ == 'nsubj':
                    not_first.add(child.text)
                elif child.dep_ == 'neg':   # If first verb is negated throw out sequence of terms
                    is_first = False
                    break

            # If sequence of FIRST_PERSON .. VERB .. TO .. VERB is present, add it
            if is_first:
                return True
    return False


def worker_init(*props):
    """ Initialization function for Pool workers """
    global parser
    parser = load('en_core_web_md')

    global not_first
    not_first = set()


def tag_document(props):
    """ Parses document then checks for intent """
    index, context = props

    # Parse context
    parsed = parser(context)

    # For basic structure in context
    return index if identify_basic_intent(parsed) else None


def tag_intent_documents(contexts):
    """ Determines whether each context contains intent, then returns a boolean mask """
    from utilities.data_management import load_execution_params

    # For document in corpus
    worker_pool = Pool(load_execution_params()['n_threads'], initializer=worker_init)
    intent_indexes = worker_pool.imap(
        tag_document,
        ((index, context) for index, context in enumerate(contexts)),
        chunksize=50
    )
    worker_pool.close()
    worker_pool.join()

    # Pull out indexes
    intent_indexes = filter(lambda index: index is not None, intent_indexes)

    # Convert intention index list to boolean array
    intent_mask = zeros(len(contexts), dtype=bool)
    intent_mask[list(intent_indexes)] = True

    print('intent percentage', sum(intent_mask) / len(contexts))

    return intent_mask


def get_intent_terms(contexts, intent_mask=None, content_data=None):
    """ Computes intention terms """
    if intent_mask is None:
        # Get contexts with intent
        intent_mask = tag_intent_documents(contexts)
        print('Mask computed, running doc matrix')

    document_matrix, features = generate_context_matrix(contexts) if content_data is None else content_data

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
    for term, lab_count, unlabelled_count in relevant_contexts:
        if unlabelled_count == 0:
            unlabelled_count = 1

        freq = (lab_count / num_labelled) / (unlabelled_count / num_unlabelled)
        significant_terms.append((term, freq))

    significant_terms = list(filter(lambda term: term[1] > 1, significant_terms))
    significant_terms = sorted(significant_terms, key=lambda term: term[1], reverse=True)

    return significant_terms, intent_mask
