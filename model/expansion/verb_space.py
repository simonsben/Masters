from utilities.data_management import split_embeddings
from numpy import max, min, asarray, all, mean, zeros
from scipy.spatial.distance import cosine
from scipy.linalg import norm
from config import training_verbosity


def get_cube_mask(embeddings, target_labels, tolerance=6):
    """ Returns the labels whose vectors are within the hyper-cube formed by the target labels """
    target_labels = set(target_labels)
    labels, vectors = split_embeddings(embeddings)

    target_mask = asarray([label in target_labels for label in labels])

    minimums = min(vectors[target_mask], axis=0)
    maximums = max(vectors[target_mask], axis=0)

    if tolerance is not None:
        divisions = maximums - minimums
        modifications = divisions * (1 + tolerance) / 2
        minimums -= modifications
        maximums += modifications

    within = asarray([
        all([vector >= minimums, vector <= maximums]) for vector in vectors
    ])

    return set(labels[within]), within


def get_cone_mask(embeddings, target_labels, tolerance=1):
    """
    Returns labels whose vectors are within the hyper-cone formed by the target labels

    :param ndarray embeddings: Numpy array with the first column containing tokens and remainder containing vectors
    :param list target_labels: List of labels to expand with the provided embeddings
    :param float tolerance: Percentage to increase bounding area parameters (percentage as decimal, default 1)
    """
    target_labels = set(target_labels)
    labels, vectors = split_embeddings(embeddings)

    target_verb_mask = asarray([label in target_labels for label in labels])

    central_vector = mean(vectors[target_verb_mask], axis=0)
    cone_angle = max([cosine(central_vector, vector) for vector in vectors[target_verb_mask]])

    target_verb_magnitudes = [norm(vector) for vector in vectors[target_verb_mask]]
    min_magnitude = min(target_verb_magnitudes)
    max_magnitude = max(target_verb_magnitudes)
    magnitude_range = (max_magnitude - min_magnitude)

    if tolerance is not None:
        cone_angle *= (1 + tolerance)

        magnitude_range *= tolerance
        min_magnitude -= magnitude_range
        max_magnitude += magnitude_range

        if min_magnitude < 0:
            min_magnitude = 0

        if training_verbosity > 0:
            print('Cone mask angle', cone_angle, 'and min max', min_magnitude, max_magnitude)

    within = zeros(vectors.shape[0], bool)
    distances = zeros(within.shape, float)
    for index, vector in enumerate(vectors):
        magnitude = norm(vector)
        distances[index] = cosine(central_vector, vector)
        within[index] = (
                min_magnitude <= magnitude <= max_magnitude and
                distances[index] <= cone_angle
        )

    return set(labels[within]), within, distances
