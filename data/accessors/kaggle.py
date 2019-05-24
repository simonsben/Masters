from numpy import array


def kaggle_accessor(document):
    """ Accessor for Kaggle dataset """
    return document[2]


def kaggle_mutator(modified_content, values, document):
    """
    Mutator for Kaggle dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    is_abusive = 1 if sum(array(document[3:]).astype(int)) > 0 else 0

    modified_document = [is_abusive] + values + [modified_content]
    return modified_document
