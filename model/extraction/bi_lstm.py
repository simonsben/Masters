from numpy import zeros
from utilities.data_management import load_dataset_params


def vectorize_content(content, model):
    """ Translates document content to word vectors """
    vector_dim = model.get_dimension()
    max_tokens = load_dataset_params()['max_document_tokens']

    prepared_content = content.split(' ')
    vector_content = zeros((max_tokens, vector_dim))

    for ind, token in enumerate(prepared_content):
        if ind >= max_tokens: break

        vector = model.get_word_vector(token)
        vector_content[ind] = vector

    return vector_content


def vectorize_data(dataset, model):
    """ Converts a dataset's content to word vectors """

    dataset['vectorized_content'] = dataset['document_content'].apply(vectorize_content, model=model)
