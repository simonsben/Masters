from utilities.data_management import make_path, load_vector, open_w_pandas, check_existence
# from utilities.plotting import scatter_plot, show
from numpy import asarray, logical_not, sum, where
from config import dataset

base = make_path('data/processed_data/') / dataset / 'analysis'
data_base = base / 'intent'
initial_label_path = data_base / 'cone_mask.csv'
round_label_gen = lambda index: data_base / ('midway_mask_%d_of_10.csv' % index)
context_path = data_base / 'contexts.csv'
english_mask_path = data_base / 'english_mask.csv'

check_existence([context_path, english_mask_path, initial_label_path])

english_mask = load_vector(english_mask_path).astype(bool)
raw_contexts = open_w_pandas(context_path).loc[english_mask]
contexts = raw_contexts['contexts'].values

initial_labels = load_vector(initial_label_path)[english_mask]
round_labels = asarray([load_vector(round_label_gen(index)) for index in range(10)]).transpose()

print(contexts.shape, round_labels.shape)


certain_start = initial_labels != .5
uncertain_start = logical_not(certain_start)
positive_end = round_labels[:, -1] >= 1
negative_end = round_labels[:, -1] <= 0

changes = initial_labels != round_labels[:, -1]

ground_changes = changes.copy()
ground_changes[uncertain_start] = False
[ground_changes] = where(ground_changes)
print(sum(ground_changes), 'changes to ground-truth labels')

for index in ground_changes:
    print(round_labels[index], contexts[index])

print(sum(positive_end[uncertain_start]), 'positive documents labelled')
print(sum(negative_end[uncertain_start]), 'negative documents labelled')

