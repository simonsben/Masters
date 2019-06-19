from utilities.data_management import open_w_pandas, make_path, move_to_root, check_existence, split_sets, \
    to_numpy_array, load_execution_params
from numpy import where
from fastText import load_model as load_fast
from model.extraction import vectorize_data
from model.training import load_attention

move_to_root()

# Define dataset paths
params = load_execution_params()
dataset_name = params['dataset']
fast_text_name = params['fast_text_model']
source_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
pred_dir = make_path('data/predictions') / dataset_name
fast_text_filename = make_path('data/lexicons/fast_text/') / (fast_text_name + '.bin')
model_path = make_path('data/models/') / dataset_name / 'derived/fast_text.h5'
train_pred_path, test_pred_path = pred_dir / 'train.csv', pred_dir / 'test.csv'

# Check existence of files
check_existence(train_pred_path)
check_existence(test_pred_path)
check_existence(model_path)
check_existence(fast_text_filename)
check_existence(source_path)

# Open predictions
train_predictions = open_w_pandas(train_pred_path, index_col=0)
test_predictions = open_w_pandas(test_pred_path, index_col=0)

# Open and split processed data
dataset = open_w_pandas(source_path)
fast_text_model = load_fast(str(fast_text_filename))
print('Raw data loaded')

processed_data = vectorize_data(dataset, fast_text_model)
fast_text_model = None
print('Vectors ready')

train_set, test_set = split_sets(processed_data)
train_set, test_set = to_numpy_array([train_set, test_set])
print('All data ready')

# Load model and make predictions
model = load_attention(model_path)
# train_predictions['fast_text'] = where(model.predict(train_set) > .5, 1, 0)
# test_predictions['fast_text'] = where(model.predict(test_set) > .5, 1, 0)
train_predictions['fast_text'] = model.predict(train_set)
test_predictions['fast_text'] = model.predict(test_set)

# Save predictions
train_predictions.to_csv(train_pred_path)
test_predictions.to_csv(test_pred_path)

# Print data summary
print(train_predictions.describe())
print(test_predictions.describe())
