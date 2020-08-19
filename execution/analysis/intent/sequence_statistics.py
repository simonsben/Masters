from utilities.data_management import make_path, make_dir, load_vector, check_existence
from utilities.plotting import show, subplots, set_labels, savefig
from numpy import loadtxt, percentile, zeros, arange, min, max, flip, argsort
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
savefig(figure_base / 'positive_sequence_rates.png')

plot_percentiles(negative_percentiles, plot_labels, 'Negative sequence rates during training', axis_labels)
savefig(figure_base / 'negative_sequence_rates.png')

num_samples = 30

for e_index, epoch in enumerate(positive):
    indexes = flip(argsort(epoch))[:num_samples]
    print('Epoch', e_index + 1)

    for s_index in indexes:
        print('\t%20s %5.2f' % (ngrams[s_index], positive[e_index, s_index]))


from numpy import asarray
mask = asarray(['going' in gram for gram in ngrams])
indexes = flip(argsort(positive[-1, mask]))[:num_samples]

print('\n' * 5)
for index in indexes:
    print('%20s %.3f %.3f' % (ngrams[mask][index], positive[-1, mask][index], negative[-1, mask][index]))

show()
