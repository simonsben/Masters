from utilities.data_management import prepare_csv_reader, make_path, check_existence
from pandas import read_csv
from numpy import array, vstack
import config

# Define params
dataset_name = config.dataset
lexicon_name = config.fast_text_model
expansion_name = 'emotions_mixed_redef'

# Define file paths
file_dir = make_path('data/processed_data') / dataset_name / 'analysis' / 'lexicon_expansion'
raw_path = file_dir / (expansion_name + '.csv')
original_path = make_path('data/prepared_lexicon/nrc_emotion_lexicon.csv')

# Check for files
check_existence(raw_path)
check_existence(original_path)

# Load original lexicon
lexicon = read_csv(original_path)
words = lexicon['word'].values
emotions = lexicon.columns[1:]

# Load expanded terms
reader, fl, header = prepare_csv_reader(raw_path)
expansion = array([new_terms for new_terms in reader])

print(lexicon.iloc[:, 1:].sum(axis=0))

print(len(expansion), len(header), len(lexicon))
# Ensure data dimensions are as expected
if len(expansion) != len(header):
    raise ValueError('Original lexicon not equal to expansion length')

# Assemble expanded lexicons
for emotion in emotions:
    emotion_mask = lexicon[emotion] > 0

    new_terms = expansion[emotion_mask]
    original_terms = words[emotion_mask]

    expanded_lexicon = vstack([original_terms, new_terms])

    print(emotion)
    print(expanded_lexicon)
