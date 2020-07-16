from model.training.rate_limiting import term_rate_limit
from numpy import around, percentile, logical_not, asarray, ndarray, sum, argsort, all, flip, log, where
from scipy.sparse import csr_matrix
from pandas import DataFrame
from config import confidence_increment, training_verbosity, prediction_threshold


def sequence_counts(document_matrix, mask):
    """ Get count of each token in document matrix under a specific mask """
    target_documents = document_matrix[mask]
    [token_sums] = asarray(target_documents.sum(axis=0))

    return token_sums


def compute_sequence_rates(positive_counts, negative_counts, num_positive_documents, num_negative_documents):
    """
    Computes the normalized rates for each sequence

    :param ndarray positive_counts: Array indicating the number of times a sequence was in positive documents
    :param ndarray negative_counts: Array indicating the number of times a sequence was in negative documents
    :param int num_positive_documents: Number of positive documents
    :param int num_negative_documents: Number of negative documents
    """
    # Make minimum number of occurrences of all sequences one in both positive and negative documents
    positive_counts[positive_counts == 0] = 1
    negative_counts[negative_counts == 0] = 1

    # Compute a 'normalizing constant' that takes into account unequal class sizes
    normalizing_constant = log(num_positive_documents) / log(num_negative_documents)

    # Compute normalized sequence rates
    positive_rates = normalizing_constant * (positive_counts / negative_counts)
    negative_rates = positive_rates ** -1

    return positive_rates, negative_rates


def get_significant_tokens(token_frequencies, target_column, threshold=prediction_threshold):
    """
    Get array of significant tokens for a set of given frequencies

    :param DataFrame token_frequencies: DataFrame of tokens and their corresponding document frequencies
    :param int target_column: Index of column of interest
    :param float threshold: Percentile threshold to deep terms *significant*
    :return ndarray: Array of significant tokens
    """
    frequencies = token_frequencies.values[:, target_column]                # Extract relevant frequencies
    threshold_value = max(25, percentile(frequencies[frequencies > 1], threshold))   # Compute threshold value

    significance_mask = frequencies > threshold_value                       # Compute mask of values above threshold
    [index_map] = where(significance_mask)

    subset_indexes = flip(argsort(frequencies[significance_mask]))
    return index_map[subset_indexes]


def train_sequence_learner(current_labels, sequences, document_matrix):
    """
    Identifies significant token n-grams to current labels and computes a new set of labels

    :param ndarray current_labels: Array of current intent labels
    :param list sequences: List of token n-grams listed in the document matrix
    :param csr_matrix document_matrix: Sparse document matrix
    :return: intent tokens, non-intent tokens tokens, updated labels
    """

    # Get subset of non uncertain data to use for training
    useful_mask = current_labels != .5
    training_matrix = document_matrix[useful_mask]

    positive_mask = around(current_labels[useful_mask]).astype(bool)  # Get mask for examples of positive intent
    negative_mask = logical_not(positive_mask)                        # Get mask for examples of negative intent

    # Get number of occurrences of tokens in positive, negative, and uncertain documents
    positive_count = sequence_counts(training_matrix, positive_mask)
    negative_count = sequence_counts(training_matrix, negative_mask)

    # token_totals = positive_count + negative_count + uncertain_count
    num_positive_documents = sum(positive_mask)
    num_negative_documents = sum(negative_mask)

    positive_rates, negative_rates = compute_sequence_rates(positive_count, negative_count,
                                                            num_positive_documents, num_negative_documents)

    sequence_rates = {'token': sequences, 'positive': positive_rates, 'negative': negative_rates}
    token_frequencies = DataFrame(sequence_rates)

    # Get significant tokens
    positive_indexes = get_significant_tokens(token_frequencies, 1)
    negative_indexes = get_significant_tokens(token_frequencies, 2)

    document_matrix = document_matrix.tocsc()
    positive_matrix = document_matrix[:, positive_indexes]
    negative_matrix = document_matrix[:, negative_indexes]

    packed_data = term_rate_limit(positive_matrix, negative_matrix, current_labels)
    has_intent_terms, has_non_intent_terms, intent_index, non_intent_index = packed_data

    no_intent_terms = logical_not(has_intent_terms)
    no_non_intent_terms = logical_not(has_non_intent_terms)

    # Get mask of documents that have supporting and no contradicting tokens and are unfrozen
    has_intent = all([has_intent_terms, no_non_intent_terms], axis=0)
    has_non_intent = all([no_intent_terms, has_non_intent_terms], axis=0)

    # Modify current mask
    return_mask = current_labels.copy()
    return_mask[has_intent] += confidence_increment
    return_mask[has_non_intent] -= confidence_increment

    # Bound labels to [0, 1]
    return_mask[return_mask < 0] = 0
    return_mask[return_mask > 1] = 1

    if training_verbosity > 0:
        print('Term learner changes', sum(has_intent) + sum(has_non_intent))

        print('Round features')
        print(sequences[positive_indexes][:intent_index])
        print(sequences[negative_indexes][:non_intent_index])

    return return_mask
