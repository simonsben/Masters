from numpy import ndarray, vectorize, histogram, cumsum, argmin, sqrt, asarray, all
from empath import Empath


def compute_norm(value_one, value_two, norm=2):
    return (value_one ** norm + value_two ** norm) ** (1 / norm)


def compute_abusive_intent(intent_predictions, abuse_predictions, use_distribution=True):
    """ Compute a 'score' for abusive intent from intent and abuse predictions """
    if not isinstance(intent_predictions, ndarray):
        raise TypeError('Expected intent predictions to be a numpy array.')
    if not isinstance(abuse_predictions, ndarray):
        raise TypeError('Expected abuse predictions to be a numpy array.')
    if intent_predictions.shape != abuse_predictions.shape:
        raise TypeError('Intent predictions and abuse predictions must be the same length.')
    if len(intent_predictions.shape) > 1:
        raise TypeError('Predictions should be a vector, not an array')

    if use_distribution:
        cumulative_function = vectorize(estimate_cumulative(intent_predictions))

        return abuse_predictions * cumulative_function(intent_predictions)

    norm = vectorize(compute_norm)
    abusive_intent = norm(intent_predictions, abuse_predictions)

    return abusive_intent


# TODO correct to space bins based on distribution
def estimate_cumulative(data, num_bins=150):
    """
    Estimates the cumulative distribution of a dataset
    :param data: Data vector, numpy array
    :param num_bins: Number of bins to use in the estimation, int
    :return: Estimated function
    """
    distribution, bin_edges = histogram(data, bins=num_bins)
    bin_edges = bin_edges[:-1]

    cumulative = cumsum(distribution)
    cumulative -= cumulative[0]
    cumulative = sqrt(cumulative / cumulative[-1])

    def cumulative_function(prediction):
        relative_locations = bin_edges <= prediction
        if relative_locations[-1]:              # If its in the last bin
            return cumulative[-1]

        bin_index = argmin(relative_locations)  # Get index of the bin
        return cumulative[bin_index]            # Return approx cumulative sum at the point

    return cumulative_function


def estimate_joint_cumulative(data_a, data_b, resolution=.01):
    """
    Estimates an (independent) joint distribution of two datasets
    :param data_a: Data vector, numpy array
    :param data_b: Data vector, numpy array
    :param resolution: Resolution of estimation
    :return: Estimated function
    """
    cumulative_function_a = estimate_cumulative(data_a, num_bins=int(1 / resolution * 2))
    cumulative_function_b = estimate_cumulative(data_b, num_bins=int(1 / resolution * 2))

    cumulative_function_a = vectorize(cumulative_function_a)
    cumulative_function_b = vectorize(cumulative_function_b)

    def join_cumulative_function(prediction_a, prediction_b):
        return cumulative_function_a(prediction_a) * cumulative_function_b(prediction_b)

    return join_cumulative_function


def get_verbs(raw_frames, column_index, unique=True):
    """ Extracts the verbs from an intent frame matrix """
    raw_verbs = raw_frames[:, column_index]
    verbs = raw_verbs[raw_verbs != '']

    if unique:
        verb_set = {}
        for verb in verbs:
            verb_set[verb] = 1 + (verb_set[verb] if verb in verb_set else 0)
        verb_set = sorted(
            [(verb, verb_set[verb]) for verb in verb_set],
            key=lambda _set: _set[1],
            reverse=True
        )
        verb_set = list(map(lambda _set: _set[0], verb_set))

        return verb_set
    return verbs


sample_categories = ['kill', 'leisure', 'exercise', 'communication']


def get_polarizing_mask(tokens, categories=sample_categories):
    """ Get a mask of polarizing/strong action verbs """
    thing = Empath()
    is_polarizing = asarray([
        sum(thing.analyze(verb, categories=categories).values())
        for verb in tokens
    ]) != 0

    return is_polarizing


def intent_verb_filename(name, model_name):
    """ Generates the filename for intent verb embeddings """
    return name + '_vectors-' + model_name + '.csv.gz'


def refine_mask(mask, tokens, document_tokens, token_index=None):
    tokens = set(tokens).copy()

    if token_index is not None:
        document_tokens = document_tokens[:, token_index]

    correction_mask = asarray([token not in tokens for token in document_tokens])

    mask = mask.copy()
    mask[all([correction_mask, mask == 1], axis=0)] = .5

    return mask
