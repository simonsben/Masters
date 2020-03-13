from utilities.data_management import split_embeddings
from numpy import max, min, asarray, all, mean, zeros
from scipy.spatial.distance import cosine
from scipy.linalg import norm


def get_cube_mask(embeddings, target_labels, tolerance=3):
    target_labels = set(target_labels)
    labels, vectors = split_embeddings(embeddings)

    target_mask = asarray([label in target_labels for label in labels])

    minimums = min(vectors[target_mask], axis=0)
    maximums = max(vectors[target_mask], axis=0)

    if tolerance is not None:
        division = maximums - minimums
        modifications = division * (1 + tolerance) / 2
        minimums -= modifications
        maximums += modifications

    within = asarray([
        all([vector >= minimums, vector <= maximums]) for vector in vectors
    ])

    return set(labels[within]), within


def get_cone_mask(embeddings, target_labels, tolerance=1):
    target_labels = set(target_labels)
    labels, vectors = split_embeddings(embeddings)

    target_mask = asarray([label in target_labels for label in labels])

    central_vector = mean(vectors[target_mask], axis=0)
    cone_angle = max([cosine(central_vector, vector) for vector in vectors[target_mask]])

    magnitudes = [norm(vector) for vector in vectors[target_mask]]
    min_magnitude = min(magnitudes)
    max_magnitude = max(magnitudes)
    magnitude_range = max_magnitude - min_magnitude

    if tolerance is not None:
        cone_angle *= (1 + tolerance)

        magnitude_range *= tolerance
        min_magnitude -= magnitude_range
        max_magnitude += magnitude_range

        if min_magnitude < 0:
            min_magnitude = 0

    within = zeros(vectors.shape[0], bool)
    for index, vector in enumerate(vectors):
        magnitude = norm(vector)
        within[index] = (
                min_magnitude < magnitude < max_magnitude and
                cosine(central_vector, vector) <= cone_angle
        )

    return set(labels[within]), within
