from numpy import ndarray, zeros_like, vectorize, histogram, cumsum, argmin, sqrt


def compute_norm(value_one, value_two, norm=2):
    return (value_one ** norm + value_two ** norm) ** (1 / norm)


def compute_abusive_intent(intent_predictions, abuse_predictions, use_multiplication=True):
    """ Compute a 'score' for abusive intent from intent and abuse predictions """
    if not isinstance(intent_predictions, ndarray):
        raise TypeError('Expected intent predictions to be a numpy array.')
    if not isinstance(abuse_predictions, ndarray):
        raise TypeError('Expected abuse predictions to be a numpy array.')
    if intent_predictions.shape != abuse_predictions.shape:
        raise TypeError('Intent predictions and abuse predictions must be the same length.')
    if len(intent_predictions.shape) > 1:
        raise TypeError('Predictions should be a vector, not an array')

    if use_multiplication:
        return intent_predictions * abuse_predictions

    norm = vectorize(compute_norm)
    abusive_intent = norm(intent_predictions, abuse_predictions)

    return abusive_intent


# TODO correct to space bins based on distribution
def estimate_cumulative(predictions, num_bins=150):
    distribution, bin_edges = histogram(predictions, bins=num_bins)
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
