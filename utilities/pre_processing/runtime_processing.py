from re import compile
from numpy import zeros

# Run clean on contexts
non_char = compile(r'[^a-zA-Z ]')    # Replace non-alphabetic characters
extra_space = compile(r'\s{2,}')     # Replace repeat spaces
repeats = compile(r'(.)(\1{2,})')
acronym = compile(r'(\w\.){2,}')
split_pattern = compile(r'[.?!;]+')


def clean_acronym(document):
    """ Removes periods from acronyms (ex. U.S.A. -> USA) """
    return acronym.sub(lambda match: match[0].replace('.', '') + ' ', document)


def pre_intent_clean(document):
    """ Perform final clean on contexts before saving and exporting. """
    document = repeats.sub(lambda match: match[0][0], document)     # Remove repeat characters
    return extra_space.sub(' ', document)                          # Remove extra spaces


def final_clean(document):
    """ Final clean after context splitting """
    return extra_space.sub(' ', non_char.sub(' ', document))     # Run post clean regex


def simulated_runtime_clean(documents):
    """ Simulates the cleaning process from partial to saved contexts """
    for index, document in enumerate(documents):
        if not isinstance(document, str):
            documents[index] = ''
            continue

        documents[index] = extra_space.sub(' ', non_char.sub(
            ' ',
            pre_intent_clean(clean_acronym(document))
        ))

    return documents


def runtime_clean(documents):
    """ Last function to be executed before training (mainly used when executing on Google Colab) """
    for ind, document in enumerate(documents):
        if not isinstance(document, str):
            documents[ind] = ''
            continue

        documents[ind] = extra_space.sub(' ', non_char.sub(' ', document))

    return documents


def token_to_index(raw_documents, raw_words, max_tokens):
    """ Takes documents and replaces their tokens with the index within the word embeddings """
    words = {term: index for index, term in enumerate(raw_words)}
    indexed_documents = []

    for document in raw_documents:
        if not isinstance(document, str): continue

        indexed_document = []
        for token_index, token in enumerate(document.split(' ')):
            if token_index > max_tokens: break
            if len(token) == 0:
                continue
            elif token in words:
                indexed_document.append(words[token])

        indexed_documents.append(indexed_document)

    doc_arrays = zeros((len(indexed_documents), max_tokens))
    for ind, document in enumerate(indexed_documents):
        num_tokens = min([max_tokens, len(document)])
        doc_arrays[ind, :num_tokens] = document[:num_tokens]

    return doc_arrays
