from utilities.data_management import make_path, check_existence, open_w_pandas, get_model_path, vector_to_file
from model.networks import predict_abusive_intent
from config import dataset


embedding_path = make_path('data/models/') / dataset / 'derived' / (dataset + '.bin')
processed_base = make_path('data/processed_data') / dataset / 'analysis'
context_path = processed_base / 'intent' / 'contexts.csv'
predictions_base = processed_base / 'intent_abuse'
intent_weights_path = get_model_path('intent')
abuse_weights_path = get_model_path('abuse')

check_existence([embedding_path, context_path, intent_weights_path, abuse_weights_path])
print('Config complete.')

raw_contexts = open_w_pandas(context_path)['contexts'].values
abuse, intent, abusive_intent = predict_abusive_intent(raw_contexts)

vector_to_file(abuse, predictions_base / 'abuse_predictions.csv')
vector_to_file(intent, predictions_base / 'intent_predictions.csv')
vector_to_file(abusive_intent, predictions_base / 'abuse_intent_predictions.csv')
print('Complete.')
