from warnings import warn

# Data constants
dataset = 'dataset_name'
fast_text_model = 'model_name'

# Execution platform constants
n_threads = 4

max_token_values = {
    'dataset_name': 200
}
embedding_dimensions = {
    'model_name': 200
}

max_tokens = max_token_values[dataset]
embedding_dimension = embedding_dimensions[fast_text_model]

warn('Loaded execution params with dataset %s and fastText model %s' % (dataset, fast_text_model), RuntimeWarning)
