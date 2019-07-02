def stormfront_accessor(document):
    """ Accessor for the stormfront dataset """
    return document[5].encode('ascii', 'xmlcharrefreplace').decode('ascii')


def stormfront_mutator(modified_content, values, document):
    """
    Mutator for the stornfront dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    is_abusive = ''

    modified_document = [values[0], is_abusive] + values[1:] + [modified_content]
    return modified_document
