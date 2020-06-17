from unidecode import unidecode


def stormfront_accessor(document):
    """ Accessor for the stormfront dataset """
    return document[-1]


def stormfront_mutator(modified_content, values, document):
    """
    Mutator for the stornfront dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    user = unidecode(document[-2]) if isinstance(document[-2], str) else 'NO_VALID_USERNAME'

    modified_document = [values[0], user] + values[1:] + [modified_content]
    return modified_document
