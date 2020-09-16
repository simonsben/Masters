from warnings import warn

# Data constants
dataset = 'dataset'
fast_text_model = 'model_name'

# Plotting constants
font_size = 12

# Execution platform constants
n_threads = 0

max_token_values = {
    'dataset': 0,
}

embedding_dimensions = {
    'model_name': 0
}

max_tokens = max_token_values[dataset]
embedding_dimension = embedding_dimensions[fast_text_model]

# Deep learning constants
training_verbosity = 1
execute_verbosity = 1
batch_size = 512
confidence_increment = .1
prediction_threshold = 0.99
sequence_threshold = 0.999
num_training_rounds = 20
mask_refinement_method = 'cone'

warn('Loaded execution params with dataset %s and fastText model %s' % (dataset, fast_text_model), RuntimeWarning)
