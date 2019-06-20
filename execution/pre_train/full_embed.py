from utilities.data_management import load_execution_params, check_existence, move_to_root, make_path, open_w_pandas, \
    open_fast_embed, check_writable
from fastText import load_model
from numpy import zeros, column_stack
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
index = 0
for word in lex[0]:
    words[str(word)] = index
    index += 1

oov_count = 0
oov_index = index
for doc in data['document_content']:
    for word in doc.split(' '):
        if str(word) not in words:
            oov_count += 1

            words[str(word)] = index
            index += 1

print(round(oov_count / index * 10000) / 100, '% of words out of lexicon')

# Define data structure for embedding vectors
embeddings = DataFrame(columns=(['words'] + list(range(0, 300))))
embeddings['words'] = sorted(words.keys())

fast_model = load_model(str(mod_path))
print('Model loaded, completing embeddings')

for ind, word in enumerate(embeddings['words'].values):
    tmp = words[word]
    if tmp < oov_index:
        embeddings.iloc[ind, 1:] = lex.iloc[tmp, 1:].values
    else:
        embeddings.iloc[ind, 1:] = fast_model.get_word_vector(word)

print('All embeddings generated, saving')

embeddings.to_csv(dest_path)
