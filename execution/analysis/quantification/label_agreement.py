from utilities.data_management import make_path, check_existence, open_w_pandas
from numpy import zeros, zeros_like, logical_not
from model.analysis.labelling import enforce_qualifying

label_map = {
    'NEGATIVE': 0,
    'POSITIVE': 1
}
index_key = 'context_id'
intent_key, abuse_key = 'intent_label', 'abuse_label'
qualifying_answers = [True, True, True, False, False]

source_path = make_path('data/datasets/data_labelling/')
context_path = source_path / 'contexts.csv'
label_path = source_path / 'labels.csv'

check_existence([context_path, label_path])
print('Config complete.')

# Load data
contexts = open_w_pandas(context_path, index_col=None)
raw_labels = open_w_pandas(label_path)
print('Data loaded.')

intent_index = raw_labels.columns == intent_key
abuse_index = raw_labels.columns == abuse_key

# Remove SKIP labels
skip_mask = raw_labels[intent_key].values != 'SKIP'
labels = raw_labels.loc[skip_mask]

# Map to boolean values
labels.loc[:, intent_index] = labels[intent_key].apply(lambda label: label_map[label])
labels.loc[:, abuse_index] = labels[abuse_key].apply(lambda label: label_map[label])

# print(labels.pivot_table(values=intent_key, index=index_key, columns=intent_key))

print(labels[intent_key])

is_qualifying = labels[index_key].values < 0

enforce_qualifying(labels.loc[is_qualifying], labels.loc[logical_not(is_qualifying)], qualifying_answers, 'intent_label')

# offset = labels[index_key].min()
# intent_sums = zeros((labels.shape[0], 2), dtype=int)
# abuse_sums = zeros_like(intent_sums)
#
# for label in labels[[index_key, intent_key, abuse_key]].values:
#     index, intent_label, abuse_label = label
#     intent_sums[index + offset] += int(intent_label)
#     abuse_sums[index + offset] += int(abuse_label)
#
# print(intent_sums)
