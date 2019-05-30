from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, \
    get_path_maps, load_xgboost_model, match_feature_weights
from utilities.analysis import get_feature_values
from matplotlib.pyplot import show
from utilities.plotting import feature_significance
from os import mkdir
from numpy import load

move_to_root(4)

# Define file paths
dataset_name = '24k-abusive-tweets'
model_dir = make_path('data/models/') / dataset_name
figure_dir = make_path('figures/') / dataset_name / 'derived' / 'feature_significance/'
data_dir = make_path('data/processed_data/') / dataset_name / 'derived'

# Check for files
if not figure_dir.exists():
    mkdir(figure_dir)

# Import data
maps = get_path_maps()
dir_maps = maps['layer_class']
name_maps = maps['layer_names']

layers = ['char_n_grams', 'othering', 'word_n_grams']

# Get model and feature values
for layer in layers:
    model_path = model_dir / dir_maps[layer] / (layer + '.bin')
    check_existence(model_path)

    feature_path = data_dir / (layer + '_terms.npy')
    check_existence(feature_path)

    layer_features = load(feature_path, allow_pickle=True)
    model = load_xgboost_model(model_path)
    weights = get_feature_values(model)
    gains = get_feature_values(model, 'gain')

    feature_weights = match_feature_weights(layer_features, weights)
    feature_gains = match_feature_weights(layer_features, gains)

    # Stacked model
    feature_significance(feature_weights, name_maps[layer] + ' Weights',
                         filename=figure_dir / (layer + '_weight.png'))
    feature_significance(feature_gains, name_maps[layer] + ' Gains', is_weight=False, x_log=True,
                         filename=figure_dir / (layer + '_gain.png'))

# show()
