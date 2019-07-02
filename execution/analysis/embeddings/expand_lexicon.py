from model.analysis.expand_lexicon import expand_lexicon
from utilities.data_management import move_to_root, make_path, check_existence, load_execution_params, \
    prepare_csv_writer, make_dir
from dask.dataframe import read_csv
from pandas import read_csv as pandas_read

# Lexicon expansion
# Takes a initial lexicon and expands it using word embeddings
# Exported as a .csv with the first row as the original lexicon, and the second being the expanded one

# Define paths
move_to_root(4)

params = load_execution_params()
embed_name = params['fast_text_model']
data_name = params['dataset']
lexicon_name = 'desire'

lexicon_path = make_path('data/lexicons/intent/') / (lexicon_name + '.csv')
embed_path = make_path('data/prepared_lexicon/') / (embed_name + '.csv')
dest_path = make_path('data/processed_data/') / data_name / 'analysis' / 'lexicon_expansion' / (lexicon_name + '.csv')

check_existence(embed_path)
check_existence(lexicon_path)
make_dir(dest_path)


# Define dataset-specific constants
dtypes = {str(ind): float for ind in range(1, 301)}
dtypes[0] = str

# Import data
embeddings = read_csv(embed_path, dtype=dtypes)
lexicon = pandas_read(lexicon_path, header=None)[0].values
print('Data imported')

print(lexicon)

# Expand lexicon
# lexicon = ['bitch', 'fuck', 'idiot']
expanded = expand_lexicon(lexicon, embeddings)

num_terms = sum([len(w_expanded) for w_expanded in expanded])
print('Lexicon expanded from', len(lexicon), 'to', num_terms, 'terms')

# Save data
writer, fl = prepare_csv_writer(dest_path)
writer.writerow(lexicon)
for w_expanded in expanded:
    writer.writerow(w_expanded)

fl.close()
