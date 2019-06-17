from utilities.data_management import open_w_pandas, make_path, check_existence, check_writable, move_to_root, \
    save_prepared, load_execution_params
from model.extraction import hurtlex, subjectivity
from model.training import train_xg_boost

move_to_root()

# Define source files
dataset_name = load_execution_params()['dataset']
data_filename = make_path('data/prepared_data/') / (dataset_name + '.csv')
lexicon_base = make_path('data/prepared_lexicon/')
processed_base = make_path('data/processed_data/') / dataset_name / 'lexicon'
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
model_dir = make_path('data/models/' + dataset_name + '/lexicon')
check_writable(model_dir)
check_writable(processed_base)

# Check lexicons
for layer in sub_layers:
    check_existence(lexicon_base / (layer['lexicon_name'] + '.csv'))

# Load dataset
dataset = open_w_pandas(data_filename)
print('Data loaded.')

# Train models
for layer in sub_layers:
    model_name = layer['model_name']
    model_filename = model_dir / (model_name + '.bin')
    if model_filename.exists():
        print('Skipping', model_name)
        continue

    print('Starting', model_name)

    # Load lexicon and construct document-term matrix
    lexicon = open_w_pandas(lexicon_base / (layer['lexicon_name'] + '.csv'))
    document_matrix = layer['executor'](dataset, lexicon)

    # Train model
    model, (train, test) = train_xg_boost(document_matrix, dataset['is_abusive'], return_data=True)

    # Save model
    model.save_model(str(model_filename))
    save_prepared(processed_base, model_name, train[0], test[0])
    print(model_name, 'completed.')
