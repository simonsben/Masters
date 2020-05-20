from numpy import all, asarray, logical_not, ndarray, any


def get_consensus(current_labels, *labels, confidence=.2):
    """
    Ensures consensus between label sources and produces new array of labels

    :param ndarray current_labels: Current array of labels
    :param float confidence: Amount to change the modification [default .2]
    :return ndarray: New array of labels with changes applied
    """
    if len(labels) < 1:
        raise AttributeError('No labels provided to get consensus.')
    elif len(labels) < 2:
        ResourceWarning('Only provided a single list of labels, so consensus has no function.')
        return labels[0]

    labels = asarray(labels)

    has_positive = any(labels > .5, axis=0)
    has_negative = any(labels < .5, axis=0)

    no_conflict = logical_not(
        all([has_positive, has_negative], axis=0)
    )
    positive = all([no_conflict, has_positive], axis=0)
    negative = all([no_conflict, has_negative], axis=0)

    current_labels = current_labels.copy()
    current_labels[positive] += confidence
    current_labels[negative] -= confidence

    return current_labels
