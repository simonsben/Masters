from re import compile, findall, subn

hashtag_regex = compile(r'#[a-zA-Z0-9_]+')
hashtag_parser_regex = compile(r'[a-z]+|[A-Z][a-z]+|[A-Z]+(?![a-z])|\d+')


def split_hashtags(document):
    """ Identifies hashtags and splits them, where possible """
    def replace(_match):
        hashtag = _match.group(0)
        bits = [bit.lower() for bit in findall(hashtag_parser_regex, hashtag)]
        print(hashtag, ' '.join(bits))

        return ' '.join(bits)

    document, num_hashtags = subn(hashtag_regex, replace, document)

    if num_hashtags > 0:
        print(document, num_hashtags)

    return num_hashtags, document
