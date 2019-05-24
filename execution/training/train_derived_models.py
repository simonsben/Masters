from utilities.data_management import open_w_pandas, make_path, check_existence, check_writable
from model.extraction import n_gram_matrix, othering_matrix
from model.training import train_xg_boost

# Define source files
data_filename = '../../data/prepared_data/24k-abusive-tweets.csv'
lexicon_path = make_path('../../data/prepared_lexicon/nrc_emotion_lexicon.csv')
check_existence(lexicon_path)

sub_layers = [
    {
        'model_name': 'othering',
        'executor': othering_matrix
    },
    {
        'model_name': 'word_n_grams',
        'executor': n_gram_matrix
    },
    {
        'model_name': 'char_n_grams',
        'executor': lambda doc: n_gram_matrix(doc, 5000, False)
    }
]

# Define destination directory
model_dir = make_path('../../data/models/derived/')
check_writable(model_dir)

# Load dataset
dataset = open_w_pandas(data_filename)
print('Data loaded.')

for layer in sub_layers:
    print('Starting ', layer['model_name'])

    # Train model
    document_matrix = layer['executor'](dataset)
    model, [test_data, test_labels] \
        = train_xg_boost(document_matrix, dataset['is_abusive'], return_test=True)

    # Save model
    model_filename = str(model_dir / (layer['model_name'] + '.mod'))
    model.save_model(model_filename)
    print(layer['model_name'], ' completed.')
