def hannah_data_accessor(document):
    """ Accessor for Hannah's dataset """
    return document[1]


def hannah_data_mutator(modified_content, values, document):
    """
    Mutator for Hannah's dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    is_abusive = document[0]

    modified_document = [values[0], is_abusive] + values[1:] + [modified_content]
    return modified_document
