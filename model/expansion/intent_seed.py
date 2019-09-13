from spacy import load
from numpy import asarray, squeeze, logical_not, add, percentile, sum
from itertools import compress
from multiprocessing import Pool
from model.extraction import generate_context_matrix

first_person = {'i', 'we', 'me', 'us', 'em', 'mine', 'myself', 'ourselves'}
good_verbs = {'VB', 'VBG', 'VBP', 'VBZ'}
alt_question_indicators = {'if'}


def identify_basic_intent(context):
    """ Determines if parsed document contains a sequence of term that indicate intent """
    parsed = parser(context)

    # For each token in parsed document
    for token in parsed:
        if token.tag_ != 'TO':  # Start check from TO
            continue
        elif token.head.pos_ != 'VERB' or token.head.head.pos_ != 'VERB':  # If there aren't two verbs, pass
            continue

        base_verb = token.head.head  # Get base verb
        # Check for past tense verb
        if base_verb.tag_ not in good_verbs:
            continue

        # Check for negation or question
        if len([
            tok for tok in base_verb.children
            if tok.dep_ == 'neg' or tok.tag_ == 'WRB' or tok.text in alt_question_indicators
        ]) > 0:
            del parsed
            return 0

        # Check for at least one related personal pronoun
        tmp = [tok.text for tok in base_verb.children if tok.tag_ == 'PRP']
        if len(tmp) < 1:
            continue
        elif len([tok for tok in tmp if tok in first_person]) > 0:
            del parsed
            return 1

        # Check for at least one future or conditional verb modifier
        if len([tok for tok in base_verb.children if tok.dep_ == 'aux' and tok.tag_ == 'MD']) > 0:
            del parsed
            return 1
        del parsed
        return .9

    del parsed
    return .5


def worker_init(*props):
    """ Initialization function for Pool workers """
    global parser
    parser = load('en_core_web_md')


def tag_intent_documents(contexts):
    """ Determines whether each context contains intent, then returns a boolean mask """
    from utilities.data_management import load_execution_params

    # Initialize worker pool
    worker_pool = Pool(load_execution_params()['n_threads'], initializer=worker_init)

    # Process documents
    intent_values = worker_pool.map(identify_basic_intent, contexts)

    # Close pool
    worker_pool.close()
    worker_pool.join()

    print('intent percentage', sum(intent_values == 1) / len(intent_values))

    return asarray(intent_values)


def get_intent_terms(contexts, intent_values=None, content_data=None):
    """ Computes intention terms """
    if intent_values is None:
        # Get contexts with intent
        intent_values = tag_intent_documents(contexts)
        print('Mask computed, running doc matrix', intent_values)

    intent_mask = intent_values == 1
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

    return significant_terms, intent_values
