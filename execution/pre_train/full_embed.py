from utilities.data_management import load_execution_params, check_existence, move_to_root, make_path, open_w_pandas, \
    open_fast_embed, check_writable
from fastText import load_model
from pandas import DataFrame

move_to_root()

params = load_execution_params()
lex_name = params['fast_text_model']
data_name = params['dataset']

lex_base = make_path('data/lexicons') / 'fast_text'
lex_path = lex_base / (lex_name + '.vec')
mod_path = lex_base / (lex_name + '.bin')
data_path = make_path('data/prepared_data') / (data_name + '.csv')
dest_path = make_path('data/prepared_lexicon/') / (lex_name + '.csv')

check_existence(lex_path)
check_existence(mod_path)
check_existence(data_path)
check_writable(dest_path)
print('Paths defined, starting')

data = open_w_pandas(data_path)
lex = open_fast_embed(lex_path)
lex.drop(columns=301, inplace=True)
print('Data imported')

words = {}
for ind, word in enumerate(lex[0]):
    words[str(word)] = lex.iloc[ind, 1:].values


fast_model = load_model(str(mod_path))
print('Model loaded, generating oov vectors')

oov_embed = {}
for doc in data['document_content']:
    for word in doc.split(' '):
        if str(word) not in words:
            oov_embed[str(word)] = fast_model.get_word_vector(str(word))

print(round(len(oov_embed) / len(words) * 10000) / 100, '% of words out of lexicon')


# Define data structure for embedding vectors
full_set = {**words, **oov_embed}
print('Merged dictionaries, generating list')

full_set = [[word] + list(full_set[word]) for word in full_set]
print('Generated list, converting to dataframe')

embeddings = DataFrame(full_set)
embeddings.sort_values(0, inplace=True)
embeddings.rename({0: 'words'}, inplace=True)
print(embeddings)

print('Dataframe complete, saving')

embeddings.to_csv(dest_path)
