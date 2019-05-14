from re import compile, subn

emoji_regex = compile(r'&#\d+;')


def count_upper(document):
    """ Counts number of uppercase characters and converts to lowercase """
    count = sum(1 for ch in document if ch.isupper())
    document = document.lower()

    return count, document


def count_emojis(document):
    """ Counts the number of emojis in the document and removes them """
    document, count = emoji_regex.subn('', document)
    return count, document


def original_length(document):
    """ Gives the length of the original document """
    return len(document), document
