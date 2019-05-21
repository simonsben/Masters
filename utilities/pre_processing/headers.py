# Constants related to document pre-processing

base_headers = [
    'is_abusive',
    'document_content'
]


def generate_header(filters):
    """ Generates the headers for the pre-processed document """
    filter_headers = [_filter(None, True) for _filter in filters if _filter(None, True) is not None]

    headers = base_headers[:1] + filter_headers + [base_headers[1]]
    return headers
