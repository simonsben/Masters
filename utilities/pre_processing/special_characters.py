from re import compile, subn, match


special_regex = compile(r'&\S+;')
special_regex_parser = compile(r'(&amp;)|(&lt;|&le;)|(&gt;|&ge;)|(&160;)|(&201;)')
group_map = {
    0: 'and',
    1: 'less than',
    2: 'greater than',
    3: ' ',
    4: 'e'
}


def manage_special_characters(document):
    """ Locates special characters and intelligently replaces (or removes) them """
    def replace(_match):
        character = _match.group(0)
        parsed = match(special_regex_parser, character)
        if parsed is not None:
            group_id = get_symbol_index(parsed.groups())
            return group_map.get(group_id)

        print(character)
        return _match.group(0)

    document, num_special = subn(special_regex, replace, document)

    return num_special, document


def get_symbol_index(groups):
    for ind, val in enumerate(groups):
        if val is not None:
            return ind
    raise ValueError('Group does not exist')
