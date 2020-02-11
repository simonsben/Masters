from utilities.data_management import load_execution_params, check_existence, move_to_root, make_path, open_w_pandas, \
    check_writable
from fasttext import load_model
from pandas import DataFrame
from utilities.pre_processing import simulated_runtime_clean

move_to_root()

# Load execution parameters
params = load_execution_params()
lex_name = params['fast_text_model']
data_name = params['dataset']

# Define paths
lexicon_base = make_path('data/lexicons/fast_text')
model_path = lexicon_base / (lex_name + '.bin')
destination_path = make_path('data/prepared_lexicon/') / (data_name + '-' + lex_name + '.csv')
data_path = make_path('data/prepared_data') / (data_name + '_partial.csv')

# Ensure paths are valid
check_existence(model_path)
check_existence(data_path)
check_writable(destination_path)
print('Paths defined, starting')

# Load data
content = open_w_pandas(data_path, index_col=None).values[:, -1]
content = simulated_runtime_clean(content)
print('Data imported')

# Load fast text model
fast_model = load_model(str(model_path))
print('Model loaded, generating word vectors')

# Generate missing embeddings
embeddings = {}
usage_counts = {}
for doc in content:
    if not isinstance(doc, str):
        continue

    for word in doc.split(' '):
        tmp_word = str(word)
        if tmp_word not in embeddings:
            embeddings[tmp_word] = fast_model.get_word_vector(tmp_word)
            usage_counts[tmp_word] = 1
        else:
            usage_counts[tmp_word] += 1

print(len(embeddings), 'word embeddings calculated')

# Convert to list
embeddings = [[word, usage_counts[word]] + list(embeddings[word]) for word in embeddings]
print('Generated list, converting to DataFrame')

# Convert to DataFrame
headings = ['words', 'usages'] + [str(ind) for ind in range(1, fast_model.get_dimension() + 1)]
fast_model = None
embeddings = DataFrame(embeddings, columns=headings)

# Sort
embeddings.sort_values(['usages', 'words'], inplace=True, ascending=[False, True])
embeddings.drop(columns='usages', inplace=True)

print(embeddings)
print('DataFrame complete, saving')

# Save word embeddings
embeddings.to_csv(destination_path, index=False)
