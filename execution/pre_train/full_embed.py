from utilities.data_management import load_execution_params, check_existence, move_to_root, make_path, open_w_pandas, \
    check_writable
from fastText import load_model
from pandas import DataFrame
from re import compile

move_to_root()

non_char = compile(r'[^a-zA-Z]')
extra_space = compile(r'[ ]+')

# Load execution parameters
params = load_execution_params()
lex_name = params['fast_text_model']
data_name = params['dataset']
partial = False
context_run = False

# Define paths
lex_base = make_path('data/lexicons') / 'fast_text'
mod_path = lex_base / (lex_name + '.bin')
dest_path = make_path('data/prepared_lexicon/') / (data_name + '-' + lex_name + ('_min' if partial else '') + '.csv')

if context_run:
    data_path = make_path('data/processed_data') / data_name / 'analysis' / 'intent' / 'contexts.csv'
else:
    data_path = make_path('data/prepared_data') / (data_name + '.csv')

# Ensure paths are valid
check_existence(mod_path)
check_existence(data_path)
check_writable(dest_path)
print('Paths defined, starting')

# Load data
content = open_w_pandas(data_path, index_col=None)['contexts' if context_run else 'document_content'].values

if context_run:
    for ind, context in enumerate(content):
        if not isinstance(context, str):
            content[ind] = ''
            continue
        # content[ind] = extra_space.sub(
        #     ' ', non_char.sub(' ', context
        #                       )
        # )
print('Data imported')

# Load fast text model
fast_model = load_model(str(mod_path))
print('Model loaded, generating oov vectors')

# Generate missing embeddings
embeddings = {}
usage_counts = {}
for doc in content:
    if not isinstance(doc, str):
        continue
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
fast_model = None
embeddings = DataFrame(embeddings, columns=headings)

if partial:
    threshold = 1
    # threshold = percentile(embeddings['usages'].values, 5)
    embeddings = embeddings.loc[embeddings['usages'] > threshold]
    print('Removed embeddings for less than', threshold, 'occurrences')

embeddings.sort_values(['usages', 'words'], inplace=True, ascending=[False, True])
embeddings.drop(columns='usages', inplace=True)
print(embeddings)

print('Dataframe complete, saving')

embeddings.to_csv(dest_path, index=False)
