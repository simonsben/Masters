from unidecode import unidecode


def iron_march_accessor(document):
    """ Accessor for the stormfront dataset """
    return document[2]


def iron_march_mutator(modified_content, values, document, user_index=-3):
    """
    Mutator for the Iron March dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    user = document[user_index] if isinstance(document[user_index], int) else 'NO_VALID_USERNAME'

    modified_document = [values[0], user] + values[1:] + [modified_content]
    return modified_document
