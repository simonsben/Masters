from numpy import ndarray, vectorize, histogram, cumsum, argmin, sqrt, asarray, all
from empath import Empath


sample_categories = ['kill', 'leisure', 'exercise', 'communication']


def compute_norm(component_one, component_two, norm=2):
    """ Vector norm, euclidean norm by default """
    return (component_one ** norm + component_two ** norm) ** (1 / norm)


def compute_abusive_intent(intent_predictions, abuse_predictions, method='product'):
    """
    Compute a 'score' for abusive intent from intent and abuse predictions

    :param ndarray intent_predictions: Array of intent predictions
    :param ndarray abuse_predictions: Array of abuse predictions
    :param str method: Choice of abusive intent computation
    :return ndarray: Array of abusive intent predictions
    """
    if not isinstance(intent_predictions, ndarray):
        raise TypeError('Expected intent predictions to be a numpy array.')
    if not isinstance(abuse_predictions, ndarray):
        raise TypeError('Expected abuse predictions to be a numpy array.')
    if intent_predictions.shape != abuse_predictions.shape:
        raise TypeError('Intent predictions and abuse predictions must be the same length.')
    if len(intent_predictions.shape) > 1:
        raise TypeError('Predictions should be a vector, not an array')

    if method == 'cdf':
        cumulative_function = vectorize(estimate_cumulative(intent_predictions))
        return abuse_predictions * cumulative_function(intent_predictions)
    elif method == 'euclidean':
        norm = vectorize(compute_norm)
        return norm(intent_predictions, abuse_predictions)
    elif method == 'product':
        pass
    else:
        UserWarning('Invalid method choice, using product')

    return intent_predictions * abuse_predictions


# TODO correct to space bins based on distribution
def estimate_cumulative(data, num_bins=1000):
    """
    Estimates the cumulative distribution of a dataset

    :param ndarray data: Data vector, numpy array
    :param int num_bins: Number of bins to use in the estimation, int
    :return: Estimated function
    """
    distribution, bin_edges = histogram(data, bins=num_bins)
    bin_edges = bin_edges[:-1]

    cumulative = cumsum(distribution)           # Get cumulative sum
    cumulative -= cumulative[0]                 # Shift distribution to align with bin edges
    cumulative = cumulative / cumulative[-1]    # Convert cumulative to percentile

    def cumulative_function(prediction):
        relative_locations = bin_edges <= prediction
        if relative_locations[-1]:              # If its in the last bin
            return cumulative[-1]

        bin_index = argmin(relative_locations) - 1  # Get index of the bin
        return cumulative[bin_index]            # Return approx cumulative sum at the point

    return cumulative_function


def estimate_joint_cumulative(data_a, data_b, resolution=0.001):
    """
    Estimates an (independent) joint distribution of two datasets

    :param ndarray data_a: Data vector, numpy array
    :param ndarray data_b: Data vector, numpy array
    :param float resolution: Resolution of estimation
    :return: Estimated function
    """
    cumulative_function_a = estimate_cumulative(data_a, num_bins=int(1 / resolution * 2))
    cumulative_function_b = estimate_cumulative(data_b, num_bins=int(1 / resolution * 2))

    cumulative_function_a = vectorize(cumulative_function_a)
    cumulative_function_b = vectorize(cumulative_function_b)

    def join_cumulative_function(prediction_a, prediction_b):
        return sqrt(cumulative_function_a(prediction_a) * cumulative_function_b(prediction_b))

    return join_cumulative_function


def get_verbs(raw_frames, column_index, unique=True):
    """
    Extracts the verbs from an intent frame matrix

    :param ndarray raw_frames: Array containing intent frames
    :param int column_index: Index of verb column
    :param bool unique: Whether to return verbs or unique verbs
    """
    raw_verbs = raw_frames[:, column_index]     # Get verbs
    verbs = raw_verbs[raw_verbs != '']          # Remove zero length verbs, if present

    # If unique verbs are requested
    if unique:
        # Collect set of verbs with usage counts in dictionary
        verb_set = {}
        for verb in verbs:
            verb_set[verb] = 1 + (verb_set[verb] if verb in verb_set else 0)

        # Sort terms by decreasing frequency
        verb_set = sorted(
            [(verb, verb_set[verb]) for verb in verb_set],
            key=lambda _set: _set[1],
            reverse=True
        )
        verbs = list(map(lambda _set: _set[0], verb_set))
    return verbs


def get_polarizing_mask(tokens, categories=None):
    """ Compute mask of tokens present in *polarizing* Empath categories """
    if categories is None:
        categories = sample_categories

    # Initialize empath
    empath = Empath()

    # Construct mask of tokens present in polarizing *categories* within Empath
    is_polarizing = asarray([
        sum(                                                        # Take sum of presence over categories
            empath.analyze(token, categories=categories).values()   # Check whether token is in polarizing categories
        ) for token in tokens
    ]) != 0                                                         # Mask of non-zero presence in polarizing categories

    return is_polarizing


def refine_rough_labels(rough_labels, refined_tokens, document_tokens, token_index=None):
    """
    Refine positive rough labels based on refined set of intent tokens

    :param ndarray rough_labels: Array of rough labels
    :param list refined_tokens: List of refined tokens (that indicate *strong* intent)
    :param ndarray document_tokens: Array of tokens for each document
    :param int token_index: Index of tokens within document_tokens array [optional]
    :return ndarray: Refined rough labels
    """
    refined_tokens = set(refined_tokens).copy()     # Compute set of unique tokens

    # If token index is present, use to select column from document tokens array
    if token_index is not None:
        document_tokens = document_tokens[:, token_index]
    elif len(document_tokens.shape) > 1:
        raise TypeError('Document token array is multi-dimensional, but no token index was passed.')

    # Check whether the document token is in the refined set
    correction_mask = asarray([token not in refined_tokens for token in document_tokens])

    # Apply refinements to labels where current label is positive
    refined_labels = rough_labels.copy()
    refined_labels[all([correction_mask, refined_labels == 1], axis=0)] = .5

    return refined_labels
