from utilities.data_management import move_to_root, open_w_pandas, make_path, check_existence, get_path_maps, \
    load_xgboost_model, match_feature_weights
from utilities.analysis import get_feature_values
from matplotlib.pyplot import show, close
from utilities.plotting import feature_significance, shap_feature_significance
from os import mkdir
from scipy.sparse import load_npz
import config

move_to_root(4)

layer_names = ['hurtlex', 'subjectivity', 'empath']

# Define file paths
dataset_name = config.dataset
model_dir = make_path('data/models/') / dataset_name
figure_base = make_path('figures/') / dataset_name / 'lexicon'
fig_dir = figure_base / 'feature_significance/'
shap_dir = figure_base / 'shap'
lexicon_dir = make_path('data/prepared_lexicon/')
dataset_dir = make_path('data/processed_data') / dataset_name / 'lexicon'

# Check for files
if not fig_dir.exists(): mkdir(fig_dir)
if not shap_dir.exists(): mkdir(shap_dir)

# Import data
maps = get_path_maps()
dir_maps = maps['layer_class']
name_maps = maps['layer_names']

# Get model and feature values
for layer in layer_names:
    model_path = model_dir / dir_maps[layer] / (layer + '.bin')
    lexicon_path = lexicon_dir / maps['lexicon_names'][layer]
    dataset_path = dataset_dir / (layer + '_test.npz')

    check_existence(model_path)
    check_existence(lexicon_path)

    features = open_w_pandas(lexicon_path, index_col=0)['word'].values
    dataset = load_npz(dataset_path)

    model = load_xgboost_model(model_path)
    weights = get_feature_values(model)
    gains = get_feature_values(model, 'gain')
    t_gains = get_feature_values(model, 'total_gain')

    feature_weights = match_feature_weights(features, weights)
    feature_gains = match_feature_weights(features, gains)
    feature_t_gains = match_feature_weights(features, t_gains)

    # Stacked model
    feature_significance(feature_weights, name_maps[layer] + ' Weights',
                         filename=fig_dir / (layer + '_weight.png'))
    feature_significance(feature_gains, name_maps[layer] + ' Gains', is_weight=False, x_log=True,
                         filename=fig_dir / (layer + '_gain.png'))
    feature_significance(feature_t_gains, name_maps[layer] + ' Total Gains', is_weight=False, x_log=True,
                         filename=fig_dir / (layer + '_t_gain.png'))
    shap_feature_significance(model, dataset, name_maps[layer] + ' SHAP Weights', features=features,
                              filename=shap_dir / (layer + '.png'))

# show()
close('all')
