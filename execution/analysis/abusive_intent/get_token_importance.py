from utilities.data_management import read_csv, move_to_root, make_path, load_execution_params, check_existence, \
    make_dir
from model.networks import generate_abuse_network, generate_intent_network
from utilities.plotting import plot_token_importance, show
from utilities.pre_processing import runtime_clean, token_to_index

move_to_root(4)
params = load_execution_params()
abuse_run = False
run_name = 'abuse' if abuse_run else 'intent'
max_tokens = 250 if abuse_run else 100
lexicon = 'abusive_intent-mixed_redef' if abuse_run else 'storm-front-fast_text'

dataset = params['dataset']
# lexicon = params['fast_text_model']

weight_path = make_path('data/models/') / dataset / 'analysis' / (run_name + '_model_weights.h5')
lexicon_path = make_path('data/prepared_lexicon') / (lexicon + '.csv')
data_base = make_path('data/processed_data/') / dataset / 'analysis'
analysis_base = data_base / 'intent_abuse'
figure_base = make_path('figures/') / dataset / 'analysis' / 'abusive_intent'

check_existence(lexicon_path)
check_existence(weight_path)
make_dir(figure_base)
print('Config complete.')

# Load model
raw_embeddings = read_csv(lexicon_path)
embeddings = raw_embeddings.values[:, 1:]
tokens = raw_embeddings['words'].values
print('Embeddings loaded.')

if abuse_run:
    deep_model = generate_abuse_network(embeddings, max_tokens, True)
else:
    deep_model = generate_intent_network(embeddings, max_tokens, True)

deep_model.load_weights(weight_path)
print('Model loaded.')

# Load data
high_labels = read_csv(analysis_base / 'high_indexes.csv', header=None)[0].values
low_labels = read_csv(analysis_base / 'low_indexes.csv', header=None)[0].values
raw_contexts = read_csv(data_base / 'intent' / 'contexts.csv')['contexts'].values

contexts = runtime_clean(raw_contexts)
indexed_contexts = token_to_index(contexts, tokens)
print('Data loaded and processed.')

path_function = lambda index: figure_base / (run_name + '_token_shap_' + str(index) + '.png')

plot_token_importance(contexts, indexed_contexts, high_labels, deep_model, path_function)
print('Computed shap values')

show()
