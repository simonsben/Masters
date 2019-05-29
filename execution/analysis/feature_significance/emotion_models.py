from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, check_writable, \
    get_path_maps, load_xgboost_model, match_feature_weights
from utilities.analysis import get_feature_values
from matplotlib.pyplot import show
from utilities.plotting import feature_significance
from os import mkdir
from numpy import logical_not, isnan

move_to_root(4)

# Define file paths
dataset_name = '24k-abusive-tweets'
model_dir = make_path('data/models/') / dataset_name
figure_dir = make_path('figures/') / dataset_name / 'emotion' / 'feature_significance/'
lexicon_path = make_path('data/prepared_lexicon/nrc_emotion_lexicon.csv')

# Check for files
if not figure_dir.exists():
    mkdir(figure_dir)

# Import data
maps = get_path_maps()
dir_maps = maps['layer_class']
name_maps = maps['layer_names']

lexicon = open_w_pandas(lexicon_path, index_col=0)
features = lexicon.index.values


# Get model and feature values
for layer in lexicon.columns:
    model_path = model_dir / dir_maps[layer] / (layer + '.bin')

    check_existence(model_path)

    layer_features = features[logical_not(isnan(lexicon[layer].values))]

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

show()
