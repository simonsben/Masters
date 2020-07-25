from utilities.data_management import make_path, make_dir, load_vector, open_w_pandas, check_existence
from utilities.plotting import scatter_plot, show, subplots, set_labels
from numpy import loadtxt, percentile, zeros, arange, min, max
from config import dataset

base = make_path('data/processed_data') / dataset / 'analysis' / 'intent'
positive_path = base / 'positive_sequence_rates.csv'
negative_path = base / 'negative_sequence_rates.csv'
sequence_path = base / 'ngrams.csv'
figure_base = make_path('figures') / dataset / 'analysis'

check_existence([positive_path, negative_path, sequence_path])
make_dir(figure_base)

ngrams = load_vector(sequence_path)
positive = loadtxt(positive_path, delimiter=',', dtype=float)
negative = loadtxt(negative_path, delimiter=',', dtype=float)

percentiles = [99.999, 99.99, 99.9, 99]

positive_percentiles = zeros((positive.shape[0], len(percentiles)))
negative_percentiles = zeros((positive.shape[0], len(percentiles)))
for index, target_percentile in enumerate(percentiles):
    positive_percentiles[:, index] = percentile(positive, target_percentile, axis=1)
    negative_percentiles[:, index] = percentile(negative, target_percentile, axis=1)

plot_labels = [str(p) for p in percentiles]


def plot_percentiles(values, labels, title, ax_labels):
    x = arange(positive_percentiles.shape[0])

    fig, ax = subplots()
    ax.plot(x, values)
    ax.set_xlim(min(x), max(x))

    ax.legend(labels)
    set_labels(ax, title, ax_labels)


axis_labels = ('Epoch', 'Sequence rate')

plot_percentiles(positive_percentiles, plot_labels, 'Positive sequence rates during training', axis_labels)
plot_percentiles(negative_percentiles, plot_labels, 'Negative sequence rates during training', axis_labels)

show()
