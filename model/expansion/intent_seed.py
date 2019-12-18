from spacy import load
from numpy import asarray, squeeze, logical_not, add, percentile, sum
from itertools import compress
from multiprocessing import Pool
from model.extraction import generate_context_matrix

first_person = {'i', 'we', 'me', 'us', 'em', 'mine', 'myself', 'ourselves'}
good_verbs = {'VB', 'VBG', 'VBP', 'VBZ'}
past_verbs = {'VBD', 'VBN'}
alt_question_indicators = {'if', 'do'}


def identify_basic_intent(context):
    """ Determines if parsed document contains a sequence of term that indicate intent """
    parsed = parser(context)
    base_verb = None

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
        elif base_verb.tag_ in past_verbs:
            return 0, base_verb.text

        # Check for negation or question
        for tok in base_verb.children:
            # If token is not a negation, question, or an alternate question indicator
            if tok.dep_ != 'neg' and tok.tag_ != 'WRB' and tok.text not in alt_question_indicators:
                continue

            return 0, base_verb.text

        # Check for at least one related personal pronoun
        tmp = [tok for tok in base_verb.children if tok.text in first_person]
        if len(tmp) < 1:
            continue

        return 1, base_verb.text
    return .5, base_verb.text if base_verb is not None else None


def worker_init(*props):
    """ Initialization function for Pool workers """
    global parser
    parser = load('en_core_web_md')


def tag_intent_documents(contexts):
    """ Determines whether each context contains intent, then return intent value and base verb """
    from utilities.data_management import load_execution_params

    # Initialize worker pool
    worker_pool = Pool(load_execution_params()['n_threads'], initializer=worker_init, maxtasksperchild=50000)

    # Process documents
    intent_data = worker_pool.map(identify_basic_intent, contexts)

    # Close pool
    worker_pool.close()
    worker_pool.join()

    # Split intent values and base verbs
    intent_data = asarray(intent_data)
    intent_values = intent_data[:, 0].astype(float)
    base_verbs = intent_data[:, 1]

    print('intent percentage', sum(intent_values == 1) / len(intent_values))

    return intent_values, base_verbs


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
