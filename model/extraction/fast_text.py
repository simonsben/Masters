from numpy import zeros


def vectorize_content(content, model):
    """ Translates document content to word vectors """
    prepared_content = content.split(' ')
    vector_dim = model.get_dimension()
    vector_content = zeros((len(prepared_content), vector_dim))

    for ind, token in enumerate(prepared_content):
        vector = model.get_word_vector(token)
        vector_content[ind] = vector

    return vector_content


def vectorize_data(dataset, model):
    """ Converts a dataset's content to word vectors """

    vectorizer = lambda document: vectorize_content(document, model)
    dataset['vectorized_content'] = dataset['document_content'].apply(vectorizer)
