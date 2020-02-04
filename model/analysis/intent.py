from numpy import ndarray, zeros_like, vectorize
from scipy.linalg import norm
from math import sqrt


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
