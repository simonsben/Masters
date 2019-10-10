from sklearn.feature_extraction.text import CountVectorizer
from utilities.data_management import make_path, move_to_root, load_execution_params, open_w_pandas
from scipy.sparse import save_npz
from numpy import savetxt

move_to_root()

# Load execution parameters
params = load_execution_params()
dataset = params['dataset']

# Define paths
base = make_path('data/processed_data/') / dataset / 'analysis' / 'intent'
source = base / 'contexts.csv'
dm_path = base / 'document_matrix.npz'
feat_path = base / 'ngrams.csv'

# Load contexts
contexts = open_w_pandas(source)['contexts'].values
print('Data loaded')

# Initialize vectorizer
vectorizer = CountVectorizer(ngram_range=(3, 6), max_features=25000, token_pattern=r'\b\w+\b')

# Compute context-term matrix
document_matrix = vectorizer.fit_transform(contexts)
print('Context ngram matrix computed, saving')

# Save data
save_npz(dm_path, document_matrix)
savetxt(feat_path, vectorizer.get_feature_names(), fmt='%s', delimiter=',')
print('Save complete')
