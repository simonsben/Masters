from utilities.data_management import open_w_pandas, make_path, check_existence, split_sets
from numpy import sum, equal
import config

# Load execution parameters
dataset_name = config.dataset

# Define file paths
pred_base = make_path('data/predictions/') / dataset_name
pred_path = pred_base / 'test.csv'
fast_text_path = pred_base / 'fast_text.csv'
raw_path = make_path('data/prepared_data') / (dataset_name + '.csv')

# Ensure files exist
check_existence(pred_path)
check_existence(raw_path)

# Load data
predictions = open_w_pandas(pred_path)
data = open_w_pandas(raw_path)

# Prepare to calculate accuracy
cols = predictions.columns.values
_, test_data = split_sets(data)
test_data = test_data['is_abusive'].values
threshold = .5

# Calculate accuracy for each predictor
for col in cols:
    accuracy = sum(
        equal(test_data, predictions[col].values > threshold)
    ) / test_data.shape[0] * 100

    print('|', col, '|', accuracy, '|')
