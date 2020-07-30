from utilities.data_management import make_path, load_vector, open_w_pandas, get_prediction_path, check_existence, \
    output_aggregated_abusive_intent, make_dir
from model.analysis import group_document_predictions
from numpy import argsort, unique, flip
from pandas import DataFrame
from config import dataset

# Define paths
base = make_path('data/processed_data/') / dataset / 'analysis'
context_path = base / 'intent' / 'contexts.csv'
abuse_path = get_prediction_path('abuse')
intent_path = get_prediction_path('intent')
abusive_intent_path = get_prediction_path('abusive_intent')
data_path = base / 'intent_abuse' / 'document_aggregation.csv'

check_existence([abuse_path, intent_path, abusive_intent_path, context_path])
make_dir(data_path)
print('Config complete.')

# Load data and remove wikipedia contexts
contexts = open_w_pandas(context_path)

not_wiki = contexts['document_index'].values >= 0
sorted_indexes = argsort(contexts.index.values[not_wiki])

contexts = contexts.loc[not_wiki].iloc[sorted_indexes]
abuse = load_vector(abuse_path)[not_wiki][sorted_indexes]
intent = load_vector(intent_path)[not_wiki][sorted_indexes]
print('Data loaded.')

document_indexes = contexts['document_index'].values
unique_document_indexes = unique(contexts['document_index'].values)
content = contexts['contexts'].values

# Aggregate documents and predicted values
maximum, documents = group_document_predictions(abuse, intent, content, document_indexes)
average, _ = group_document_predictions(abuse, intent, content, document_indexes, method='average')
windowed, _ = group_document_predictions(abuse, intent, content, document_indexes, method='window')
print('Grouped.')

# Identify peak aggregated values
to_output = 25
standard_indexes = flip(argsort(maximum))[:to_output]
windowed_indexes = flip(argsort(windowed))[:to_output]

# Output data to console
print('\n\n\nStandard')
output_aggregated_abusive_intent(standard_indexes, maximum, documents)

print('\n\n\nWindowed')
output_aggregated_abusive_intent(windowed_indexes, windowed, documents)

# Save data
data = DataFrame({'document_indexes': unique_document_indexes, 'maximum': maximum, 'windowed': windowed})
data.to_csv(data_path)

print(data)
