from pandas import DataFrame


def not_string(document):
    return not isinstance(document, str)


def generate_embeddings(documents, model):
    """ Generate embeddings for provided documents """
    # Initialize dictionaries
    embeddings = {}
    usage_counts = {}

    # Generate embeddings for non-empty documents
    for document in filter(not_string, documents):
        # For each token in document
        for token in document.split(' '):
            token = str(token)

            if token not in embeddings:
                embeddings[token] = model.get_word_vector(token)
                usage_counts[token] = 1
            else:
                usage_counts[token] += 1

    # Convert embedding dictionary to a double list
    embeddings = [
        [word, usage_counts[word]] + list(embeddings[word])
        for word in embeddings
    ]
    print('Generated list, converting to DataFrame')

    # Convert double list to Pandas DataFrame
    headings = ['words', 'usages'] + [str(ind) for ind in range(1, model.get_dimension() + 1)]
    embeddings = DataFrame(embeddings, columns=headings)

    # Sort embeddings by usage then drop usage column
    embeddings.sort_values(['usages', 'words'], inplace=True, ascending=[False, True])
    embeddings.drop(columns='usages', inplace=True)

    return embeddings


# def load_trained_parameters(network, weights):
