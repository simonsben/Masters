from re import compile
from numpy import zeros

# Run clean on contexts
non_char = compile(r'[^a-zA-Z]')
extra_space = compile(r'\s{2,}')


def runtime_clean(documents):
    """ Last function to be executed before training (mainly used when executing on Google Colab) """
    for index, document in enumerate(documents):
        if not isinstance(document, str):
            documents[index] = ''
            continue

    return documents


def token_to_index(raw_documents, raw_words, max_tokens=250):
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
