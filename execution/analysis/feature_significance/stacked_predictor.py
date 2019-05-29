from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, check_writable, \
    get_path_maps, load_xgboost_model, match_feature_weights
from utilities.analysis import get_feature_values
from matplotlib.pyplot import show
from utilities.plotting import feature_significance
from os import mkdir

move_to_root(4)

# Define file paths
dataset_name = '24k-abusive-tweets'
predictions_dir = make_path('data/predictions') / dataset_name
model_path = make_path('data/models/stacked/') / (dataset_name + '.bin')
model_dir = make_path('data/models/') / dataset_name
test_path = predictions_dir / 'test.csv'
figure_dir = make_path('figures/') / dataset_name / 'derived' / 'feature_significance/'

# Check for files
check_existence(test_path)
check_existence(model_path)
if not figure_dir.exists():
    mkdir(figure_dir)

# Import data
layers = open_w_pandas(test_path, index_col=0).columns.values[:-1]

maps = get_path_maps()
dir_maps = maps['layer_class']
name_maps = maps['layer_names']

# Get model and feature values
model = load_xgboost_model(model_path)
weights = get_feature_values(model)
gains = get_feature_values(model, 'gain')

if len(weights) != len(gains) or len(gains) != len(layers):
    raise ValueError('Number of layers and feature usage metrics are not equal.')

# Stacked model
layer_weights = match_feature_weights(layers, weights)
layer_gains = match_feature_weights(layers, gains)
feature_significance(layer_weights, 'Weights', filename=figure_dir / 'stacked_weight.png')
feature_significance(layer_gains, 'Gains', is_weight=False, x_log=True, filename=figure_dir / 'stacked_gain.png')

show()
