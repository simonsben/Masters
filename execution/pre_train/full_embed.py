from utilities.data_management import load_execution_params, check_existence, move_to_root, make_path, open_w_pandas, \
    check_writable
from fastText import load_model
from pandas import DataFrame
from numpy import percentile

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
print('Data imported')

# Load fast text model
fast_model = load_model(str(mod_path))
print('Model loaded, generating oov vectors')

# Generate missing embeddings
embeddings = {}
usage_counts = {}
for doc in data['document_content']:
    for word in doc.split(' '):
        if str(word) not in embeddings:
            embeddings[str(word)] = fast_model.get_word_vector(str(word))
            usage_counts[str(word)] = 1
        else:
            usage_counts[str(word)] += 1

print(len(embeddings), 'word embeddings calculated')

# Convert to list
embeddings = [[word, usage_counts[word]] + list(embeddings[word]) for word in embeddings]
print('Generated list, converting to dataframe')

headings = ['words', 'usages'] + [str(ind) for ind in range(1, fast_model.get_dimension() + 1)]
embeddings = DataFrame(embeddings, columns=headings)

threshold = percentile(embeddings['usages'].values, 5)
embeddings = embeddings.loc[embeddings['usages'] > threshold]
print('Removed embeddings for less than', threshold, 'occurrences')

embeddings.sort_values(['usages', 'words'], inplace=True, ascending=[False, True])
embeddings.drop(columns='usages', inplace=True)
print(embeddings)

print('Dataframe complete, saving')

embeddings.to_csv(dest_path, index=False)
