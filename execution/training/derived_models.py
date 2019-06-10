from utilities.data_management import open_w_pandas, make_path, check_existence, check_writable, save_prepared, \
    move_to_root, load_execution_params
from model.extraction import n_gram_matrix, othering_matrix, adverb_matrix, document_statistics
from model.training import train_xg_boost
from numpy import save, array
from time import time

move_to_root()

# Define source files
dataset_name = load_execution_params()['dataset']
data_base = make_path('data/prepared_data/')
pre_filename = data_base / (dataset_name + '.csv')
partial_filename = data_base / (dataset_name + '_partial.csv')
lexicon_path = make_path('data/prepared_lexicon/nrc_emotion_lexicon.csv')
processed_base = make_path('data/processed_data/') / dataset_name / 'derived'
model_dir = make_path('data/models/' + dataset_name + '/derived/')

# Check files
check_existence(pre_filename)
check_existence(partial_filename)
check_existence(lexicon_path)
check_writable(processed_base)
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
    },
    {
        'model_name': 'adverbs',
        'executor': adverb_matrix,
        'dataset': partial_dataset
    },
    {
        'model_name': 'doc_stats',
        'executor': document_statistics,
        'dataset': pre_dataset
    }
]

for layer in sub_layers:
    model_filename = model_dir / (layer['model_name'] + '.bin')
    if model_filename.exists():
        print('Skipping', layer['model_name'])
        continue

    model_name = layer['model_name']
    print('Starting', model_name)

    # Train model
    start = time()
    document_matrix, features = layer['executor'](layer['dataset'])
    print('Finished generating document matrix, training xg boost model', time() - start)
    print(document_matrix.shape)

    model, (train, test) \
        = train_xg_boost(document_matrix, layer['dataset']['is_abusive'].to_numpy(), return_data=True, verb=1)
    print('Model trained.')

    # Save model
    model.save_model(str(model_filename))
    save_prepared(processed_base, model_name, train[0], test[0])
    save(processed_base / (model_name + '_terms.npy'), array(features))
    print(layer['model_name'], 'completed.')
