from model.analysis.expand_lexicon import expand_lexicon
from utilities.data_management import move_to_root, make_path, check_existence, load_execution_params, \
    prepare_csv_writer, make_dir, open_w_pandas
from dask.dataframe import read_csv
from model.extraction import get_emotion_indexes

# Lexicon expansion
# Takes a initial lexicon and expands it using word embeddings
# Exported as a .csv with the first row as the original lexicon, and the second being the expanded one

# Define paths
move_to_root(4)

params = load_execution_params()
embed_name = params['fast_text_model']
data_name = params['dataset']
lexicon_name = 'nrc_emotion_lexicon'
emotions = ['anger', 'disgust', 'fear', 'negative', 'positive', 'sadness']

lexicon_path = make_path('data/prepared_lexicon/') / (lexicon_name + '.csv')
embed_path = make_path('data/prepared_lexicon/') / (embed_name + '.csv')
dest_path = make_path('data/processed_data/') / data_name / 'analysis' / 'lexicon_expansion' / \
            ('emotions' + '_' + embed_name + '.csv')

check_existence(embed_path)
check_existence(lexicon_path)
make_dir(dest_path, 3)


# Define dataset-specific constants
dtypes = {str(ind): float for ind in range(1, 301)}
dtypes[0] = str

# Import data
embeddings = read_csv(embed_path, dtype=dtypes)
lexicon = open_w_pandas(lexicon_path, index_col=None)
lex_indexes = get_emotion_indexes(lexicon, emotions)
lexicon = lexicon['word'].iloc[lex_indexes].values
print('Data imported')

print(lexicon)

# Expand lexicon
expanded = expand_lexicon(lexicon, embeddings, simple_expand=5)

num_terms = sum([len(w_expanded) for w_expanded in expanded])
print('Lexicon expanded from', len(lexicon), 'to', num_terms, 'terms')

# Save data
writer, fl = prepare_csv_writer(dest_path)
writer.writerow(lexicon)
for w_expanded in expanded:
    writer.writerow(w_expanded)

fl.close()
