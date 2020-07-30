from numpy import ndarray, zeros_like, max, min, asarray, mean


def group_document_predictions(_abuse, _intent, contexts, document_indexes, method='max'):
    """
    Aggregate documents and predictions

    :param ndarray _abuse: Array of abuse predictions
    :param ndarray _intent: Array of intent predictions
    :param ndarray contexts: Array of contexts
    :param ndarray document_indexes: Array of document indexes for each context
    :param str method: Method to use to aggregate the documents
    :return tuple[ndarray, list]: Aggregated predictions and documents
    """
    predictions = []
    abuse, intent = [], []
    document, documents = '', []
    current_index = document_indexes[0]

    for index, document_index in enumerate(document_indexes):
        new_document = document_index != current_index
        is_last = index >= (len(document_indexes) - 1)

        if new_document or is_last:
            current_index = document_index

            predictions.append(aggregate_document(
                asarray(abuse), asarray(intent), method
            ))
            documents.append(document)

            abuse, intent = [], []
            document = ''

        abuse.append(_abuse[index])
        intent.append(_intent[index])
        a, i = abuse[-1], intent[-1]
        document += ('\n%d - %.3f, %.3f, %.3f\t%s' % (index, a, i, a * i, contexts[index]))

    return asarray(predictions), documents


def aggregate_document(abuse, intent, method='max'):
    """
    Computes the document-level prediction given an array of context predictions

    :param ndarray abuse: Array of context-level abuse predictions
    :param ndarray intent: Array of context-level intent predictions
    :param str method: Method to use to compute the document level prediction
    :return float: Aggregated value
    """
    if intent.shape != abuse.shape:
        ValueError('Passed intent and abuse arrays must be the same shape')

    if method == 'max':
        pass
    elif method == 'average':
        return mean(abuse * intent)
    elif method == 'window':
        window_intent = compute_window(intent)
        window_abuse = compute_window(abuse)
        return aggregate_document(window_intent, window_abuse)

    return max(intent * abuse)


# TODO re-write so its not slow af
def compute_window(predictions, window_size=3):
    """
    Get windowed predictions from an array of context predictions

    :param ndarray predictions: Array of predictions
    :param int window_size: Size of a window (including the central context, ex. 3 -> [LEFT, CENTER, RIGHT])
    :return ndarray: Array with the max value within the window of each value (when treated as the center)
    """
    distance = int((window_size - 1) / 2)
    window_predictions = zeros_like(predictions)

    for index, _ in enumerate(predictions):
        start = max((0, index - distance))
        end = min((len(predictions), index + distance))
        window_predictions[index] = max(predictions[start:end])

    return window_predictions
