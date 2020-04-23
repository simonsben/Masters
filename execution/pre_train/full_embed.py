from utilities.data_management import check_existence, make_path, open_w_pandas, \
    check_writable, generate_embeddings
from fasttext import load_model
from utilities.pre_processing import simulated_runtime_clean
import config

# Load execution parameters
lex_name = config.fast_text_model
data_name = config.dataset

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
fast_text_model = load_model(str(model_path))
print('Model loaded, generating word vectors')

embeddings = generate_embeddings(content, fast_text_model)
print('DataFrame complete, saving')

# Save word embeddings
embeddings.to_csv(destination_path, index=False)
