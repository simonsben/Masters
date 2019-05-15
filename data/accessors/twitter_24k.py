def twitter_24k_accessor(document):
    """ Accessor for 24k Twitter dataset """
    return document[6]


def twitter_24k_mutator(modified_content, values, document):
    """
    Mutator for 24k Twitter dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    index = document[0]
    is_abusive = 0 if int(document[5]) == 2 else 1

    modified_document = [index, is_abusive] + values + [modified_content]
    return modified_document
