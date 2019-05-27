from utilities.data_management import open_w_pandas, make_path, check_existence, check_writable
from model.extraction import n_gram_matrix, othering_matrix
from model.training import train_xg_boost

# Define source files
data_base = make_path('../../data/prepared_data/')
pre_filename = data_base / '24k-abusive-tweets.csv'
partial_filename = data_base / '24k-abusive-tweets_partial.csv'
lexicon_path = make_path('../../data/prepared_lexicon/nrc_emotion_lexicon.csv')

# Check files
check_existence(pre_filename)
check_existence(partial_filename)
check_existence(lexicon_path)

# Define destination directory
model_dir = make_path('../../data/models/derived/')
check_writable(model_dir)

# Load dataset
pre_dataset = open_w_pandas(pre_filename)
partial_dataset = open_w_pandas(partial_filename)
print('Data loaded.')

sub_layers = [
    {
        'model_name': 'othering',
        'executor': othering_matrix,
        'dataset': partial_dataset
    },
    {
        'model_name': 'word_n_grams',
        'executor': n_gram_matrix,
        'dataset': pre_dataset
    },
    {
        'model_name': 'char_n_grams',
        'executor': lambda doc: n_gram_matrix(doc, 5000, False),
        'dataset': pre_dataset
    }
]

for layer in sub_layers:
    print('Starting', layer['model_name'])

    # Train model
    document_matrix = layer['executor'](layer['dataset'])
    model, [test_data, test_labels] \
        = train_xg_boost(document_matrix, layer['dataset']['is_abusive'], return_test=True)

    # Save model
    model_filename = str(model_dir / (layer['model_name'] + '.mod'))
    model.save_model(model_filename)
    print(layer['model_name'], 'completed.')
