from utilities.data_management import make_path, check_existence
from scipy.sparse import load_npz
from utilities.plotting import show, plot_line
from numpy import asarray, cumsum, sum, sort, flip
from config import dataset

# Define paths
base_path = make_path('data/processed_data/') / dataset / 'analysis' / 'intent'
matrix_path = base_path / 'full_document_matrix.npz'
n_grams_path = base_path / 'full_ngrams.csv'

figure_path = make_path('figures/') / dataset / 'analysis' / 'cumulative_plot.png'

check_existence([matrix_path, n_grams_path])

# Load data and compute sums
matrix = load_npz(matrix_path)
sequence_sums = flip(sort(asarray(matrix.sum(axis=0)).reshape(-1)))
sequence_sums = sequence_sums[sequence_sums > 5]

# Compute cumulative distribution of sequences
total_sequences = sum(sequence_sums)
cumulative = cumsum(sequence_sums) / total_sequences

# Plot distribution
axis_labels = ('Number of unique sequences', 'Percentage of total sequence occurrences')
plot_line(cumulative, 'Cumulative distribution of unique sequences vs total sequences', figure_path, axis_labels)

show()
