from model.analysis.expand_lexicon import expand_lexicon
from utilities.data_management import make_path, check_existence, prepare_csv_writer, make_dir, open_w_pandas
from model.extraction import get_emotion_indexes
import config

# Lexicon expansion
# Takes a initial lexicon and expands it using wordnet synonyms
# Exported as a .csv with the first row as the original lexicon, and the second being the expanded one

# Define paths
embed_name = config.fast_text_model
data_name = config.dataset
lexicon_name = 'mpqa_subjectivity_lexicon'
# emotion = 'sadness'

lexicon_path = make_path('data/prepared_lexicon/') / (lexicon_name + '.csv')
dest_path = make_path('data/processed_data/') / data_name / 'analysis' / 'lexicon_expansion' / \
            (lexicon_name + '_wordnet.csv')

check_existence(lexicon_path)
make_dir(dest_path, 3)


# Define dataset-specific constants
dtypes = {str(ind): float for ind in range(1, 301)}
dtypes[0] = str

# Import data
lexicon = open_w_pandas(lexicon_path)['word'].values.astype(str)
# lex_indexes = get_emotion_indexes(lexicon, emotion)
# lexicon = lexicon['word'].iloc[lex_indexes].values
print('Data imported')

print(lexicon)

# Expand lexicon
expanded = expand_lexicon(lexicon, simple_expand=5)

num_terms = sum([len(w_expanded) for w_expanded in expanded])
print('Lexicon expanded from', len(lexicon), 'to', num_terms, 'terms')

# Save data
writer, fl = prepare_csv_writer(dest_path)
writer.writerow(lexicon)
for w_expanded in expanded:
    writer.writerow(w_expanded)

fl.close()
