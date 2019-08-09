from re import compile
from html import unescape
from unidecode import unidecode
from string import printable

special_regex = compile(r'&\S+;')
printable_characters = set(printable)


def manage_special_characters(document, get_header=False):
    if get_header: return None

    # Convert html characters to string equivalent
    document = unescape(document).replace('’', '\'')

    # Convert unicode characters
    document = unidecode(document)

    return None, document


# TODO write this 10,000x better to avoid encoding then decoding
def remove_unicode_values(documents):
    """ Takes a list of items and replaces any unicode values with the HTML entity for it (i.e. 0x080 -> &#128;) """
    if isinstance(documents, list):
        return [doc.encode('ascii', 'xmlcharrefreplace').decode('ascii') for doc in documents]
    return documents.encode('ascii', 'xmlcharrefreplace').decode('ascii')
