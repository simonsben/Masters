from numpy import around, any, squeeze, percentile, logical_not, asarray, ndarray, sum, argsort, all, flip
from itertools import compress
from functools import partial
from scipy.sparse import csr_matrix
from pandas import DataFrame


def token_counts(document_matrix, mask):
    """ Get count of each token in document matrix under a specific mask """
    target_documents = document_matrix[mask]
    [token_sums] = asarray(target_documents.sum(axis=0))

    return token_sums


def compute_token_frequency(token_info, num_positive_documents, num_negative_documents, num_total_documents):
    """ Computes the frequencies of each token in positive, negative, and all the documents """
    token, positive_count, negative_count, total_count = token_info

    positive_count = (positive_count if positive_count > 0 else 1)
    negative_count = (negative_count if negative_count > 0 else 1)

    positive_frequency = (positive_count / num_positive_documents) / (negative_count / num_negative_documents)
    negative_frequency = (negative_count / num_negative_documents) / (positive_count / num_positive_documents)
    overall_frequency = (positive_count / num_positive_documents) / (total_count / num_total_documents)

    return token, positive_frequency, negative_frequency, overall_frequency


def get_significant_tokens(token_frequencies, target_column, threshold):
    """
    Get array of significant tokens for a set of given frequencies

    :param DataFrame token_frequencies: DataFrame of tokens and their corresponding document frequencies
    :param int target_column: Index of column of interest
    :param float threshold: Percentile threshold to deep terms *significant*
    :return ndarray: Array of significant tokens
    """
    frequencies = token_frequencies.values[:, target_column]                # Extract relevant frequencies
    threshold_value = percentile(frequencies[frequencies > 1], threshold)   # Compute threshold value

    significance_mask = frequencies > threshold_value                       # Compute mask of values above threshold
    sorted_indexes = flip(argsort(frequencies[significance_mask]))          # Get sorted list of significant indexes

    return token_frequencies.values[significance_mask][sorted_indexes, 0]   # Get significant tokens


def train_term_learner(current_labels, tokens, token_mapping, document_matrix, significant_threshold=99.5,
                       label_modifier=.2, frozen_threshold=.95):
    """
    Identifies significant token n-grams to current labels and computes a new set of labels
    :param ndarray current_labels: Array of current intent labels
    :param list tokens: List of token n-grams listed in the document matrix
    :param dict token_mapping: Dictionary mapping tokens to the column index in the document matrix
    :param csr_matrix document_matrix: Sparse document matrix
    :param float significant_threshold: Percentile threshold for token to be considered significant
    :param float label_modifier: Amount to modify the document label by
    :param float frozen_threshold: Current label threshold for being *frozen* to term-learner modification
    :return intent tokens, non-intent tokens tokens, overall-intent tokens, updated labels
    """

    # Get subset of non uncertain data to use for training
    useful_mask = current_labels != .5
    training_matrix = document_matrix[useful_mask]

    positive_mask = around(current_labels[useful_mask]).astype(bool)  # Get mask for examples of positive intent
    negative_mask = logical_not(positive_mask)                          # Get mask for examples of negative intent

    # Get number of occurrences of tokens in positive, negative, and uncertain documents
    positive_count = token_counts(training_matrix, positive_mask)
    negative_count = token_counts(training_matrix, negative_mask)
    uncertain_count = token_counts(document_matrix, logical_not(useful_mask))

    token_totals = positive_count + negative_count + uncertain_count
    num_positive_documents = sum(positive_mask)
    num_negative_documents = sum(negative_mask)
    total_documents = document_matrix.shape[0]

    # Generate mask for the 50th percentile terms (i.e. get rid of 'rare' terms)
    rare_threshold = percentile(token_totals, 50)
    rare_mask = token_totals > rare_threshold

    # Assemble significant terms filtering out rare features
    significant_info = compress(zip(tokens, positive_count, negative_count, token_totals), rare_mask)
    frequency_function = partial(
        compute_token_frequency, num_positive_documents=num_positive_documents,
        num_negative_documents=num_negative_documents, num_total_documents=total_documents
    )

    # token_frequencies = [frequency_function(token_info) for token_info in significant_info]
    token_frequencies = list(map(frequency_function, significant_info))
    token_frequencies = DataFrame(token_frequencies, columns=('token', 'positive', 'negative', 'total'))

    # Get significant tokens
    positive_tokens = get_significant_tokens(token_frequencies, 1, significant_threshold)
    negative_tokens = get_significant_tokens(token_frequencies, 2, significant_threshold)
    total_tokens = get_significant_tokens(token_frequencies, 3, significant_threshold)

    # Get column masks of significant tokens
    positive_mask = [token_mapping[feature] for feature in positive_tokens]
    negative_mask = [token_mapping[feature] for feature in negative_tokens]

    # Count how many positive tokens are present in each document
    positive_token_count = asarray(document_matrix[:, positive_mask].sum(axis=1)).reshape(-1)
    negative_token_count = asarray(document_matrix[:, negative_mask].sum(axis=1)).reshape(-1)

    # Mask for documents with and without intent tokens
    has_intent_terms = positive_token_count > 0
    no_intent_terms = logical_not(has_intent_terms)

    # Mask for documents with and without non-intent tokens
    has_non_intent_terms = negative_token_count > 0
    no_non_intent_terms = logical_not(has_non_intent_terms)

    # Compute labels still uncertain enough to be modified
    unfrozen_labels = any([
        current_labels > (1 - frozen_threshold),
        current_labels < frozen_threshold], axis=0
    )

    # Get mask of documents that have supporting and no unsupporting tokens and are unfrozen
    has_intent = all([has_intent_terms, no_non_intent_terms, unfrozen_labels], axis=0)
    has_non_intent = all([no_intent_terms, has_non_intent_terms, unfrozen_labels], axis=0)

    # Modify current mask
    return_mask = current_labels.copy()
    return_mask[has_intent] += label_modifier
    return_mask[has_non_intent] -= label_modifier

    # Limit range of labels to [0, 1]
    return_mask[return_mask < 0] = 0
    return_mask[return_mask > 1] = 1

    return positive_tokens, negative_tokens, total_tokens, return_mask
