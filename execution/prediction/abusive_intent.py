from utilities.data_management import make_path, check_existence, open_embeddings, open_w_pandas, load_model_weights, \
    get_model_path, vector_to_file
from utilities.pre_processing import runtime_clean, token_to_index
from model.networks import generate_intent_network, generate_abuse_network, generate_abusive_intent_network
from config import dataset, fast_text_model, max_tokens

embedding_path = make_path('data/prepared_lexicon/') / ('abusive_intent-' + fast_text_model + '.csv.gz')
processed_base = make_path('data/processed_data') / dataset / 'analysis'
context_path = processed_base / 'intent' / 'contexts.csv'
predictions_base = processed_base / 'intent_abuse'
intent_weights_path = get_model_path('intent')
abuse_weights_path = get_model_path('abuse')

check_existence([embedding_path, context_path, intent_weights_path, abuse_weights_path])
print('Config complete.')

tokens, embeddings = open_embeddings(embedding_path)
raw_contexts = open_w_pandas(context_path)['contexts'].values[:250]
print('Loaded embeddings')

contexts = runtime_clean(raw_contexts)
context_matrix, token_map = token_to_index(contexts, tokens, True)
print('Prepared contexts.')

# intent_network = generate_intent_network(max_tokens, embedding_matrix=embeddings)
# load_model_weights(intent_network, intent_weights_path)
abusive_intent_network = generate_abusive_intent_network(max_tokens, embedding_matrix=embeddings)
load_model_weights(abusive_intent_network, intent_weights_path)
load_model_weights(abusive_intent_network, abuse_weights_path)
print(abusive_intent_network.summary())

abuse_intent_predictions = abusive_intent_network.predict(context_matrix, batch_size=512, verbose=1)
vector_to_file(abuse_intent_predictions, predictions_base / 'abuse_intent_predictions.csv')
print('Complete.')
