
manifesto = {
    'content': 0,
    'is_abusive': None
}


def manifesto_accessor(document):
    """ Accessor for the manifesto dataset """
    if len(document) < 1:
        return ''
    return document[manifesto['content']]


def manifesto_mutator(modified_content, values, document):
    """
    Mutator for the manifesto dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    modified_document = [values[0], 0] + values[1:] + [modified_content]
    return modified_document
