def wikipedia_corpus_accessor(document):
    """ Accessor for wikipedia corpus dataset """
    return document[0]


def wikipedia_corpus_mutator(modified_content, values, document):
    """
    Mutator for wikipedia corpus dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    is_abusive = 0

    modified_document = [values[0], is_abusive] + values[1:] + [modified_content]
    return modified_document
