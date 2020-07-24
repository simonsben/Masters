from utilities.data_management import make_path, load_vector, open_w_pandas, check_existence, make_dir
from utilities.plotting import stacked_plot, show, bar_plot, savefig
from numpy import asarray, sum, zeros, arange, abs
from config import dataset, confidence_increment, num_training_rounds

base = make_path('data/processed_data/') / dataset / 'analysis'
data_base = base / 'intent'
initial_label_path = data_base / 'cone_mask.csv'
round_label_gen = lambda index: data_base / ('midway_mask_%d_of_%d.csv' % (index, num_training_rounds))
context_path = data_base / 'contexts.csv'
figure_base = make_path('figures') / dataset / 'analysis'

check_existence([context_path, initial_label_path])
make_dir(figure_base)
print('Config complete.')

raw_contexts = open_w_pandas(context_path)
contexts = raw_contexts['contexts'].values

initial_labels = load_vector(initial_label_path)
round_labels = asarray([load_vector(round_label_gen(index)) for index in range(num_training_rounds)])
print('Loaded data.')

num_contexts = round_labels.shape[1]
num_possible_labels = int(1 / confidence_increment) + 1
rounds = arange(0, num_training_rounds)
shares = zeros((num_possible_labels, num_training_rounds), dtype=float)

# Compute each classes share of the labels for each round
label_values = [confidence_increment * index for index in range(num_possible_labels)]
for index, label_value in enumerate(label_values):
    shares[index] = sum(abs(round_labels - label_value) < .01, axis=1) / num_contexts

print(rounds.shape, shares.shape)

# Output *grid* of class distributions between rounds
print(('%5.1f ' * num_possible_labels) % tuple(label_values))
for round_index in range(num_training_rounds):
    print(('%5.3f ' * num_possible_labels) % tuple(shares[:, round_index]))


# Plot class distributions as a stacked area graph
class_labels = [('%.1f' % value) for value in label_values]
title = 'Change in context labels throughout training'
axis_labels = ('Training round', 'Percentage of labels')

stacked_plot(rounds, shares, class_labels, title, axis_labels, figure_base / 'label_movement.png', (12, 7))


values = sum(round_labels != .5, axis=1)
labels = [str(round + 1) for round in range(round_labels.shape[0])]

_, ax = bar_plot(values, labels, 'Number of training documents available per epoch')
ax.set_xlabel('Epoch')
savefig(figure_base / 'labels_per_round.png')

show()
