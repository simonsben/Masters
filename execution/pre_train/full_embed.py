from utilities.data_management import load_execution_params, check_existence, move_to_root, make_path, open_w_pandas, \
    open_fast_embed, check_writable
from fastText import load_model
from pandas import DataFrame

move_to_root()

# Load execution parameters
params = load_execution_params()
lex_name = params['fast_text_model']
data_name = params['dataset']

# Define paths
lex_base = make_path('data/lexicons') / 'fast_text'
lex_path = lex_base / (lex_name + '.vec')
mod_path = lex_base / (lex_name + '.bin')
data_path = make_path('data/prepared_data') / (data_name + '.csv')
dest_path = make_path('data/prepared_lexicon/') / (lex_name + '.csv')

# Ensure paths are valid
check_existence(lex_path)
check_existence(mod_path)
check_existence(data_path)
check_writable(dest_path)
print('Paths defined, starting')

# Load data
data = open_w_pandas(data_path)
lex = open_fast_embed(lex_path)
lex.drop(columns=301, inplace=True)
print('Data imported')

# Initialize dict of vectors
words = {}
for ind, word in enumerate(lex[0]):
    words[str(word)] = lex.iloc[ind, 1:].values

# Load fast text model
fast_model = load_model(str(mod_path))
print('Model loaded, generating oov vectors')

# Generate missing embeddings
oov_embed = {}
for doc in data['document_content']:
    for word in doc.split(' '):
        if str(word) not in words:
            oov_embed[str(word)] = fast_model.get_word_vector(str(word))

print(round(len(oov_embed) / (len(words) + len(oov_embed)) * 10000) / 100, '% of words out of lexicon')


# Move embeddings into a DataFrame
embeddings = {**words, **oov_embed}
print('Merged dictionaries, generating list')

embeddings = [[word] + list(embeddings[word]) for word in embeddings]
print('Generated list, converting to dataframe')

embeddings = DataFrame(embeddings)
embeddings.sort_values(0, inplace=True)
embeddings.rename({0: 'words'}, inplace=True)
print(embeddings)

print('Dataframe complete, saving', embeddings.memory_usage())

embeddings.to_csv(dest_path)
