from re import compile, sub, match

# url_regex = compile(r'http://\w*\.\w+\.\w+\S*')
url_regex = compile(r'http(s)?://(w{3}\.)?(([\w\-_]+\.)+\w+)(/[\w$\-_.+!*\'()?=]*)*'
                    r'|http:?(/){0,2}\S*$')


def pull_hyperlinks(document):
    """ Locates hyperlinks in document and removes them """
    def replace(_match):
        if _match.group(3) != 't.co':
            url = _match.group(3)
            if url is not None:
                urls.append(url)
            print(_match.group(0), _match.group(3))

        return ''

    urls = []
    document = sub(url_regex, replace, document)
    urls = ('[' + ','.join(urls) + ']') if len(urls) > 0 else ''

    return urls, document
