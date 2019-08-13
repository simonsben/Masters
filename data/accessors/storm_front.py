from unidecode import unidecode

stormfront = {
    'content': 2,
    'is_abusive': None
}


def stormfront_accessor(document):
    """ Accessor for the stormfront dataset """
    return document[stormfront['content']]


def stormfront_mutator(modified_content, values, document):
    """
    Mutator for the stornfront dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    user = unidecode(document[1])

    modified_document = [values[0], user] + values[1:] + [modified_content]
    return modified_document
