from utilities.data_management import open_w_pandas, make_path, check_existence, check_writable
from model.extraction import emotions
from model.training import train_xg_boost

# Define source files
data_filename = '../../data/prepared_data/24k-abusive-tweets.csv'
lexicon_path = make_path('../../data/prepared_lexicon/nrc_emotion_lexicon.csv')
check_existence(lexicon_path)

# Define destination directory
model_dir = make_path('../../data/models/emotion/')
check_writable(model_dir)

# Load dataset
dataset = open_w_pandas(data_filename)
print('Data loaded.')

# Load lexicon and construct document-term matrix
lexicon = open_w_pandas(lexicon_path)
document_matrices, emotions = emotions(dataset, lexicon)


for document_matrix, emotion in zip(document_matrices, emotions):
    print('Starting ', emotion)

    # Train model
    model, [test_data, test_labels] \
        = train_xg_boost(document_matrix, dataset['is_abusive'], return_test=True)

    # Save model
    model_filename = str(model_dir / (emotion + '.bin'))
    model.save_model(model_filename)
    print(emotion, ' completed.')
