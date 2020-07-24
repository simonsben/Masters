from utilities.data_management import make_path, check_existence, open_w_pandas, vector_to_file
from scipy.sparse import load_npz, save_npz
from utilities.plotting import show, plot_line
from numpy import asarray, cumsum, sum, sort, flip
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from config import dataset

# Define paths
base_path = make_path('data/processed_data/') / dataset / 'analysis' / 'intent'
context_path = base_path / 'contexts.csv'
matrix_path = base_path / 'full_document_matrix.npz'
n_grams_path = base_path / 'full_ngrams.csv'

figure_path = make_path('figures/') / dataset / 'analysis' / 'unique_sequence_rate.png'

check_existence(context_path)

if not matrix_path.exists() or not n_grams_path.exists():
    print('Computing full sequence-context matrix')

    contexts = open_w_pandas(context_path)['contexts'].values
    vectorizer = CountVectorizer(ngram_range=(3, 6), token_pattern=r'\b\w+\b')

    start = time()
    document_matrix = vectorizer.fit_transform(contexts)
    print('Completed matrix in', time() - start, 'seconds, saving.')

    sequences = asarray(vectorizer.get_feature_names())

    save_npz(matrix_path, document_matrix)
    vector_to_file(sequences, n_grams_path)
else:
    check_existence([matrix_path, n_grams_path])

    # Load data and compute sums
    document_matrix = load_npz(matrix_path)
    print('Loaded full sequence-context matrix')

sequence_sums = flip(sort(asarray(document_matrix.sum(axis=0)).reshape(-1)))
sequence_sums = sequence_sums[sequence_sums > 5]

# Compute cumulative distribution of sequences
total_sequences = sum(sequence_sums)
cumulative = cumsum(sequence_sums) / total_sequences

# Plot distribution
axis_labels = ('Number of unique sequences', 'Percentage of total sequence occurrences')
plot_line(cumulative, 'Cumulative distribution of unique sequences vs total sequences', figure_path, axis_labels).grid()

print('Sparsity', document_matrix.nnz / (document_matrix.shape[0] * document_matrix.shape[1]))

show()
