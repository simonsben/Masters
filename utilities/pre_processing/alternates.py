from re import compile, subn

bitch_regex = compile(r'bitch\w*')
handle_regex = compile(r'@\w+')


def count_bitch(document, get_header=False):
    """ Counts number occurrences of "bitch" """
    if get_header: return 'bitch_count'

    document, count = subn(bitch_regex, ' ', document)

    return count, document


def count_handles(document, get_header=False):
    """ Counts the number of twitter handles """
    if get_header: return 'handle_count'

    document, count = subn(handle_regex, ' handle ', document)

    return count, document
