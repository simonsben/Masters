from utilities.data_management import open_w_pandas, move_to_root, split_sets, make_path, check_existence, \
    check_writable, load_execution_params
from numpy import linspace, argmin, abs, argmax
from utilities.analysis import calculate_loss
from pandas import DataFrame
from sklearn.metrics import roc_curve

# TODO re-write using sklearn.metrics.roc_curve then taking the max of the tpr and fpr

move_to_root()

# Load execution parameters
params = load_execution_params()
dataset_name = params['dataset']

# Define paths
prediction_path = make_path('data/predictions/') / dataset_name / 'train.csv'
label_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
dest_path = make_path('data/predictions/') / dataset_name / 'thresholds.csv'

# Check for files
check_existence(prediction_path)
check_existence(label_path)
check_writable(dest_path)

# Load data
predictions = open_w_pandas(prediction_path)
labels, _ = split_sets(open_w_pandas(label_path)['is_abusive'])

# Calculate optimal and standard losses
min_thresholds = []
for sub_layer in predictions.columns:
    fpr, tpr, thresholds = roc_curve(labels.values, predictions[sub_layer].values)
    opt_index = argmin(abs(tpr - (1 - fpr)))
    min_thresholds.append((sub_layer, thresholds[opt_index]))

min_thresholds = DataFrame(min_thresholds, columns=['sub_layer', 'threshold'])
min_thresholds.to_csv(dest_path)

print(min_thresholds)
