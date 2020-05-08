from utilities.data_management import make_path, check_existence, open_w_pandas, vector_to_file, output_abusive_intent
from model.networks import generate_abuse_network, generate_intent_network, predict_abusive_intent
from utilities.pre_processing import simulated_runtime_clean
from fasttext import load_model
from model.layers.realtime_embedding import RealtimeEmbedding
from config import dataset, embedding_dimension, max_tokens, fast_text_model as embedding_name
from numpy import argsort


embedding_path = make_path('data/lexicons/fast_text/') / (embedding_name + '.bin')
get_weights_path = lambda target: make_path('data/models/') / dataset / 'analysis/' / (target + '_model_weights.h5')
base_path = make_path('data/processed_data/') / dataset / 'analysis'
data_path = base_path / 'intent' / 'contexts.csv'
get_prediction_path = lambda name: base_path / 'intent_abuse' / (name + '_predictions.csv')

check_existence([embedding_path, data_path, get_weights_path('intent'), get_weights_path('abuse')])

data = open_w_pandas(data_path)['contexts'].values
data = simulated_runtime_clean(data)
print('Loaded and cleaned data')

embeddings_model = load_model(str(embedding_path))
realtime_data = RealtimeEmbedding(embeddings_model, data)
print('Loaded model')

intent_network = generate_intent_network(max_tokens, embedding_dimension)
intent_network.load_weights(str(get_weights_path('intent')), by_name=True)
print(intent_network.summary())

abuse_network = generate_abuse_network(max_tokens, embedding_dimension)
abuse_network.load_weights(str(get_weights_path('abuse')), by_name=True)
print(abuse_network.summary())
print('Generated networks')

predictions = predict_abusive_intent(realtime_data, abuse_network, intent_network)

# Save predictions
vector_names = ('abuse', 'intent', 'abusive_intent')
for name, prediction_vector in zip(vector_names, predictions):
    print('Saving', name)
    vector_to_file(prediction_vector, get_prediction_path(name))
print('Complete.')


abuse, intent, abusive_intent = predictions

s_indexes = argsort(abusive_intent)
num = 25
output_abusive_intent(reversed(s_indexes[-num:]), (abusive_intent, abuse, intent), data)
