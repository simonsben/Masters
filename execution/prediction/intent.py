from utilities.data_management import move_to_root, make_path, load_execution_params, check_existence, open_w_pandas
from model.networks import generate_production_intent_network, generate_production_abuse_network, predict_abusive_intent
from utilities.pre_processing import simulated_runtime_clean
from fasttext import load_model
from numpy import vectorize

move_to_root()

params = load_execution_params()
dataset = params['dataset']
embedding_name = params['fast_text_model']
max_tokens = params['max_tokens']

embedding_path = make_path('data/lexicons/fast_text/') / (embedding_name + '.bin')
get_weights_path = lambda target: make_path('data/models/') / dataset / 'analysis/' / (target + '_model_weights.h5')
data_path = make_path('data/processed_data/') / dataset / 'analysis' / 'intent' / 'contexts.csv'

check_existence(embedding_path)
check_existence(data_path)

data = open_w_pandas(data_path)['contexts'].values
data = simulated_runtime_clean(data)
print('Loaded and cleaned data')

embeddings_model = load_model(str(embedding_path))
embeddings_dim = embeddings_model.get_dimension()
print('Loaded model')

intent_network = generate_production_intent_network(max_tokens, embeddings_dim, True)
abuse_network = generate_production_abuse_network(max_tokens, embeddings_dim, True)
print('')

intent_network.load_weights(get_weights_path('intent'), by_name=True)
abuse_network.load_weights(get_weights_path('abuse'), by_name=True)

predictions = predict_abusive_intent(data, embeddings_model, abuse_network, intent_network, max_tokens)

for thing in predictions:
    print(thing.shape)
