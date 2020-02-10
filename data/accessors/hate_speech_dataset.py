def hate_speech_dataset_accessor(document):
    """ Accessor for hate speech dataset """
    return document[-1]


def hate_speech_dataset_mutator(modified_content, values, document):
    """
    Mutator for hate speech dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    is_abusive = document[0]

    modified_document = [values[0], is_abusive] + values[1:] + [modified_content]
    return modified_document
