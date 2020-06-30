from utilities.data_management import make_path, check_existence, open_w_pandas, get_prediction_path, vector_to_file, \
    make_dir, get_embedding_path
from model.networks import predict_abusive_intent
from config import dataset

# Define file paths
embedding_path = get_embedding_path()
processed_base = make_path('data/processed_data') / dataset / 'analysis'
context_path = processed_base / 'intent' / 'contexts.csv'

# Check path existance
check_existence([embedding_path, context_path])
make_dir(get_prediction_path('abuse'))
print('Config complete.')

# Load data and make predictions
raw_contexts = open_w_pandas(context_path)['contexts'].values
abuse, intent, abusive_intent = predict_abusive_intent(raw_contexts)

# Save data
vector_to_file(abuse, get_prediction_path('abuse'))
vector_to_file(intent, get_prediction_path('intent'))
vector_to_file(abusive_intent, get_prediction_path('abusive_intent'))
print('Complete.')
