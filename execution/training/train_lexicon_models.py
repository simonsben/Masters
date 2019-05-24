from utilities.data_management import open_w_pandas, make_path, check_existence, check_writable
from model.extraction import hurtlex, subjectivity
from model.training import train_xg_boost

# Define source files
data_filename = '../../data/prepared_data/24k-abusive-tweets.csv'
lexicon_base = make_path('../../data/prepared_lexicon/')
sub_layers = [
    {
        'lexicon_name': 'hurtlex',
        'executor': hurtlex,
        'model_name': 'hurtlex'
    },
    {
        'lexicon_name': 'mpqa_subjectivity_lexicon',
        'executor': subjectivity,
        'model_name': 'subjectivity'
    }
]


# Define destination directory
model_dir = make_path('../../data/models/lexicon')
check_writable(model_dir)

# Check lexicons
for layer in sub_layers:
    check_existence(lexicon_base / (layer['lexicon_name'] + '.csv'))

# Load dataset
dataset = open_w_pandas(data_filename)
print('Data loaded.')

# Train models
for layer in sub_layers:
    model_name = layer['model_name']
    print('Starting ', model_name)

    # Load lexicon and construct document-term matrix
    lexicon = open_w_pandas(lexicon_base / (layer['lexicon_name'] + '.csv'))
    document_matrix = layer['executor'](dataset, lexicon)

    # Train model
    model, [test_data, test_labels] \
        = train_xg_boost(document_matrix, dataset['is_abusive'], return_test=True)

    # Save model
    model_filename = str(model_dir / (model_name + '.mod'))
    model.save_model(model_filename)
    print(model_name, ' completed.')
