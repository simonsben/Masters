from re import compile, subn

bitch_regex = compile(r'bitch\w*')


def count_bitch(document, get_header=False):
    """ Counts number occurrences of "bitch" """
    if get_header: return 'bitch_count'

    document, count = subn(bitch_regex, ' ', document)

    return count, document
