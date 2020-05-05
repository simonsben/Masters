from utilities.data_management import make_path, check_existence, open_w_pandas
from numpy import zeros, zeros_like, logical_not, any, sum, asarray, vectorize
from model.analysis.labelling import enforce_qualifying, count_labels
from config import dataset

label_map = {
    'SKIP': -1,
    'NEGATIVE': 0,
    'POSITIVE': 1
}

index_key = 'context_id'
intent_key, abuse_key = 'intent_label', 'abuse_label'
qualifying_answers = [True, True, True, False, False]

source_path = make_path('data/datasets/data_labelling/')
context_path = source_path / 'contexts.csv'
label_path = source_path / 'labels.csv'
dest_path = make_path('data/processed_data/') / dataset / 'analysis' / 'intent_abuse' / 'labels.csv'

check_existence([context_path, label_path])
print('Config complete.')

# Load data
contexts = open_w_pandas(context_path, index_col=None)
labels = open_w_pandas(label_path)
print('Data loaded.')

# Map to boolean values
labels[intent_key] = labels[intent_key].map(label_map)
labels[abuse_key] = labels[abuse_key].map(label_map)

# Get mask of qualifying and non-qualifying labels
is_qualifying = labels[index_key].values < 0
is_not_qualifying = logical_not(is_qualifying)

# Remove labels associated with bad qualifying answers
qualified_labels = enforce_qualifying(labels.loc[is_qualifying], labels.loc[is_not_qualifying], qualifying_answers,
                                      intent_key)

# Get labels
counted_labels = count_labels(qualified_labels, index_key, intent_key)
satisfied_labels = any(counted_labels.values[:, 1:-1] >= 3, axis=1)
print(sum(satisfied_labels), 'usable contexts.')

counted_labels.to_csv(dest_path)
