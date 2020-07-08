from utilities.data_management import make_path, load_vector, open_w_pandas, check_existence
# from utilities.plotting import scatter_plot, show
from matplotlib.pyplot import subplots, show
from numpy import asarray, logical_not, sum, where, zeros, arange, abs
from config import dataset, confidence_increment

num_rounds = 20

base = make_path('data/processed_data/') / dataset / 'analysis'
data_base = base / 'intent'
initial_label_path = data_base / 'cone_mask.csv'
round_label_gen = lambda index: data_base / ('midway_mask_%d_of_%d.csv' % (index, num_rounds))
context_path = data_base / 'contexts.csv'

check_existence([context_path, initial_label_path])
print('Config complete.')

raw_contexts = open_w_pandas(context_path)
contexts = raw_contexts['contexts'].values

initial_labels = load_vector(initial_label_path)
round_labels = asarray([load_vector(round_label_gen(index)) for index in range(num_rounds)])
print('Loaded data.')

num_contexts = round_labels.shape[1]

num_possible_labels = int(1 / confidence_increment) + 1
rounds = arange(0, num_rounds)
shares = zeros((num_possible_labels, num_rounds), dtype=float)

label_values = [confidence_increment * index for index in range(num_possible_labels)]
for index, label_value in enumerate(label_values):
    shares[index] = sum(abs(round_labels - label_value) < .01, axis=1) / num_contexts

print(rounds.shape, shares.shape)

print(('%5.1f ' * num_possible_labels) % tuple(label_values))
for round in range(num_rounds):
    print(('%5.3f ' * num_possible_labels) % tuple(shares[:, round]))

fig, ax = subplots(figsize=(10, 7))
ax.stackplot(rounds, shares, labels=[('%.1f' % value) for value in label_values])
ax.legend()
ax.grid()

show()
