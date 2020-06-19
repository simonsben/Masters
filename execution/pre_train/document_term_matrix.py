from sklearn.feature_extraction.text import CountVectorizer
from utilities.data_management import make_path, open_w_pandas, vector_to_file
from scipy.sparse import save_npz
from numpy import savetxt
import config

# Load execution parameters
dataset = config.dataset

# Define paths
base = make_path('data/processed_data/') / dataset / 'analysis' / 'intent'
source = base / 'contexts.csv'
matrix_path = base / 'document_matrix.npz'
feature_path = base / 'ngrams.csv'

# Load contexts
contexts = open_w_pandas(source)['contexts'].values.astype(str)
print('Data loaded')

# Initialize vectorizer
vectorizer = CountVectorizer(ngram_range=(3, 6), max_features=500000, token_pattern=r'\b\w+\b')

# Compute context-term matrix
document_matrix = vectorizer.fit_transform(contexts)
print('Context ngram matrix computed, saving')

# Save data
save_npz(matrix_path, document_matrix)
vector_to_file(vectorizer.get_feature_names(), feature_path)
print('Save complete')
