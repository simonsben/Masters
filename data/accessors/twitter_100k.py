def twitter_100k_accessor(document):
    """ Accessor for 100k Twitter dataset """
    return document[0]


class_map = {
    'spam': 0,
    'normal': 0,
    'abusive': 1,
    'hateful': 1
}


def twitter_100k_mutator(modified_content, values, document):
    """
    Mutator for 100k Twitter dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    is_abusive = class_map[document[1]]
    if is_abusive is None:
        raise ValueError(document[1], ' is not recognized')

    modified_document = [values[0], is_abusive] + values[1:] + [modified_content]
    return modified_document
