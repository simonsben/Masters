from utilities.data_management import open_w_pandas, make_path, split_sets, check_existence
from utilities.plotting import bar_plot
from numpy import all, sum, concatenate
from pandas import concat
# from matplotlib.pyplot import show
import config

# Define data paths
dataset_name = config.dataset
dataset_path = make_path('data/prepared_data/') / (dataset_name + '_partial.csv')
prediction_path = make_path('data/predictions/') / dataset_name / 'test.csv'
analysis_dir = make_path('data/processed_data/') / dataset_name / 'analysis'
fig_dir = make_path('figures') / dataset_name / 'analysis'

check_existence(dataset_path)
check_existence(prediction_path)

# Load data
_, dataset = split_sets(open_w_pandas(dataset_path))
predictions = open_w_pandas(prediction_path)

# Check to make sure there are the same number of rows
if len(dataset) != len(predictions):
    raise ValueError('Dataset and predictions are not the same length')

# Extract false negatives
fn_indicator = all([dataset['is_abusive'] == 1, predictions['stacked'] == 0], axis=0)
fp_indicator = all([dataset['is_abusive'] == 0, predictions['stacked'] == 1], axis=0)

false_negatives = concat([
    predictions[fn_indicator].reset_index(drop=True),
    dataset['document_content'][fn_indicator].reset_index(drop=True)
], axis=1)
false_positives = concat([
    predictions[fp_indicator].reset_index(drop=True),
    dataset['document_content'][fp_indicator].reset_index(drop=True)
], axis=1)


# Calculate number of correct indicators
pred_cols = false_negatives.columns.values[:-1]
false_negatives['good_count'] = false_negatives[pred_cols].apply(sum, axis=1)
false_positives['good_count'] = false_positives[pred_cols].apply(sum, axis=1)

# Re-order columns
cols = concatenate([pred_cols, ['good_count', 'document_content']])
false_negatives = false_negatives[cols]
false_positives = false_positives[cols]

# Save data
false_negatives.to_csv(analysis_dir / 'false_negatives.csv')
false_positives.to_csv(analysis_dir / 'false_positives.csv')

false_negatives.describe().to_csv(analysis_dir / 'fn_description.csv')
false_positives.describe().to_csv(analysis_dir / 'fp_description.csv')
predictions.describe().to_csv(analysis_dir / 'full_description.csv')

pred_mean = predictions.mean()
fn_mean = false_negatives.mean()[:-1]
fp_mean = false_positives.mean()[:-1]

bar_plot(fn_mean - pred_mean, pred_mean.index.values, 'False negative prediction residuals',
         filename=fig_dir / 'false_negative.png')
bar_plot(fp_mean - pred_mean, pred_mean.index.values, 'False positive prediction residuals',
         filename=fig_dir / 'false_positive.png')

# show()
