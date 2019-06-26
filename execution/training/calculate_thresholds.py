from utilities.data_management import open_w_pandas, move_to_root, split_sets, make_path, check_existence, \
    check_writable, load_execution_params
from numpy import linspace, argmin
from utilities.analysis import calculate_loss
from pandas import DataFrame

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
potential_thresholds = linspace(0, 1, 250)
losses = [
    [calculate_loss(predictions[col].values, labels, thresh) for thresh in potential_thresholds]
    for col in predictions.columns
]
s_losses = [calculate_loss(predictions[col].values, labels, .5) for col in predictions.columns]

min_inds = [argmin(loss) for loss in losses]
mins = DataFrame([
    (col, potential_thresholds[ind], losses[l_ind][ind], s_loss - losses[l_ind][ind])
    for l_ind, col, ind, s_loss in zip(range(len(min_inds)), predictions.columns, min_inds, s_losses)
], columns=['sub_layer', 'threshold', 'loss', 'loss_reduction'])

mins.to_csv(dest_path)

print(mins)
