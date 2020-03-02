from utilities.data_management import split_embeddings
from numpy import max, min, asarray, all, mean
from scipy.spatial.distance import cosine


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

    return set(labels[within])


def get_cone_mask(embeddings, target_labels, tolerance=.5):
    target_labels = set(target_labels)
    labels, vectors = split_embeddings(embeddings)

    target_mask = asarray([label in target_labels for label in labels])

    central_vector = mean(vectors[target_mask], axis=0)
    cone_angle = max([cosine(central_vector, vector) for vector in vectors[target_mask]])

    if tolerance is not None:
        cone_angle *= (1 + tolerance)

    within = asarray([cosine(central_vector, vector) <= cone_angle for vector in vectors])

    return set(labels[within])
