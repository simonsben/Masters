from model.extraction import generate_context_matrix
from model.expansion.intent_seed import get_intent_terms
from numpy import array, asarray, percentile
from pandas import DataFrame


def learn_terms(contexts, terms):
    """ Learns terms using a seed set of intent terms """
    # Convert terms to set
    terms = set(terms)

    # Get context matrix and context terms
    context_matrix, features = generate_context_matrix(contexts)

    # Generate feature mask
    term_mask = array([feature in terms for feature in features], dtype=bool)

    # Compute new intent mask
    intent_mask = asarray(context_matrix.multiply(term_mask).sum(axis=1)).reshape(-1)

    # Compute intent terms
    significant_terms, _ = get_intent_terms(contexts, intent_mask=intent_mask, content_data=(context_matrix, features))

    return significant_terms


def run_learning(contexts, terms, max_runs=5, threshold=50):
    """ Runs multiple rounds of term learning """
    if isinstance(terms, DataFrame):
        terms = terms['terms'].values

    num_terms = len(terms)
    print('Starting', max_runs, 'learning runs')
    print(terms)

    for run in range(max_runs):
        if isinstance(terms[0], list):
            terms = [term for term, _ in terms]

        terms = learn_terms(contexts, terms)

        # If no new terms were learned, stop
        if len(terms) == num_terms:
            break
        num_terms = len(terms)

        threshold_value = percentile([term[1] for term in terms], threshold)
        terms = list(filter(lambda term: term[1] > threshold_value, terms))

        print('Finished run', run, 'thresholded at', threshold_value)
        print(terms)

    return terms
