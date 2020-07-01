from utilities.data_management import make_path, check_existence, open_w_pandas, vector_to_file, \
    output_abusive_intent, get_embedding_path, get_prediction_path, make_dir
from model.networks import predict_abusive_intent
from numpy import argsort
from functools import partial

# Define paths
embedding_path = get_embedding_path()
base_path = make_path('data/processed_data') / 'data_labelling' / 'analysis'
data_path = base_path / 'intent' / 'contexts.csv'
prediction_path = partial(get_prediction_path, target='data_labelling')

check_existence([embedding_path, data_path])
make_dir(get_prediction_path('abuse'))
print('Config complete.')

raw_data = open_w_pandas(data_path)
data = raw_data['contexts'].values[raw_data.index.values >= 0]
print('Loaded and cleaned data')

abuse, intent, abusive_intent = predict_abusive_intent(data)

vector_to_file(abuse, prediction_path('abuse'))
vector_to_file(intent, prediction_path('intent'))
vector_to_file(abusive_intent, prediction_path('abusive_intent'))
print('Complete.')


s_indexes = argsort(abusive_intent)
num = 25
output_abusive_intent(reversed(s_indexes[-num:]), (abuse, intent, abusive_intent), data)
