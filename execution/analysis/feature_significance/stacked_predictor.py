from utilities.data_management import open_w_pandas, make_path, check_existence, get_path_maps, load_xgboost_model, \
    match_feature_weights
from utilities.analysis import get_feature_values
from matplotlib.pyplot import show, close
from utilities.plotting import feature_significance, shap_feature_significance
from os import mkdir
import config

# Define file paths
dataset_name = config.dataset
predictions_dir = make_path('data/predictions') / dataset_name
model_path = make_path('data/models/stacked/') / (dataset_name + '.bin')
model_dir = make_path('data/models/') / dataset_name
test_path = predictions_dir / 'test.csv'
fig_base = make_path('figures/') / dataset_name / 'derived'
feat_dir = fig_base / 'feature_significance/'
shap_dir = fig_base / 'shap'

# Check for files
check_existence(test_path)
check_existence(model_path)
if not feat_dir.exists(): mkdir(feat_dir)
if not shap_dir.exists(): mkdir(shap_dir)

# Import data
dataset = open_w_pandas(test_path)
dataset.drop(columns='stacked', inplace=True)
layers = dataset.columns.values

maps = get_path_maps()
dir_maps = maps['layer_class']
name_maps = maps['layer_names']

# Get model and feature values
model = load_xgboost_model(model_path)
weights = get_feature_values(model)
gains = get_feature_values(model, 'gain')
t_gains = get_feature_values(model, 'total_gain')


# Stacked model
layer_weights = match_feature_weights(layers, weights)
layer_gains = match_feature_weights(layers, gains)
layer_t_gains = match_feature_weights(layers, t_gains)

feature_significance(layer_weights, 'Weights', filename=feat_dir / 'stacked_weight.png')
feature_significance(layer_gains, 'Gains', is_weight=False, x_log=True, filename=feat_dir / 'stacked_gain.png')
feature_significance(layer_t_gains, 'Total Gains', is_weight=False, x_log=True, filename=feat_dir / 'stacked_t_gain.png')

shap_feature_significance(model, dataset, 'Stacked Predictor SHAP Weights',
                          filename=shap_dir / 'stacked.png')

# show()
close('all')
