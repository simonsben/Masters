from utilities.data_management import open_w_pandas, make_path, check_existence, check_writable, move_to_root, \
    save_prepared
from model.extraction import emotions
from model.training import train_xg_boost
import config

# Define source files
dataset_name = config.dataset
data_filename = make_path('data/prepared_data/') / (dataset_name + '.csv')
lexicon_path = make_path('data/prepared_lexicon/nrc_emotion_lexicon.csv')
processed_base = make_path('data/processed_data/') / dataset_name / 'emotion'
model_dir = make_path('data/models/' + dataset_name + '/emotion/')

# Define destination directory
check_existence(lexicon_path)
check_writable(model_dir)
check_writable(processed_base)

# Load dataset
dataset = open_w_pandas(data_filename)
print('Data loaded.')

# Load lexicon and construct document-term matrix
lexicon = open_w_pandas(lexicon_path, index_col=None)
document_matrices, emotions, features = emotions(dataset, lexicon)


for document_matrix, emotion in zip(document_matrices, emotions):
    model_filename = model_dir / (emotion + '.bin')
    if model_filename.exists():
        print('Skipping', emotion)
        continue

    print('Starting ', emotion)

    # Train model
    model, (train, test) = train_xg_boost(document_matrix, dataset['is_abusive'].to_numpy(), return_data=True)

    # Save model
    model.save_model(str(model_filename))
    save_prepared(processed_base, emotion, train[0], test[0])
    print(emotion, ' completed.')
