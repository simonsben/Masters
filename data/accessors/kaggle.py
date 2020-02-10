from numpy import asarray, sum

# id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate
attribute_mask = asarray([False, False, True, True, False, True, True, True])


def kaggle_accessor(document):
    """ Accessor for Kaggle dataset """
    return document[1]


def kaggle_mutator(modified_content, values, document):
    """
    Mutator for Kaggle dataset

    :param modified_content: Pre-processed document, string
    :param values: Extracted values, list
    :param document: Original document

    :return: Modified document, list
    """
    is_abusive = 1 if (sum(document[attribute_mask]) > 0) else 0

    modified_document = [values[0], is_abusive] + values[1:] + [modified_content]
    return modified_document
