from shap import DeepExplainer
from utilities.data_management import move_to_root, check_existence, make_path, get_path_maps, load_execution_params, \
    match_feature_weights, split_sets, open_w_pandas, to_numpy_array
from utilities.plotting import feature_significance, shap_feature_significance
from os import mkdir
from numpy import load
from pandas import read_pickle
from keras.models import load_model

move_to_root(4)

# Define file paths
# dataset_name = load_execution_params()['dataset']
dataset_name = '24k-abusive-tweets'
figure_base = make_path('figures/') / dataset_name / 'derived'
fig_dir = figure_base / 'feature_significance/'
shap_dir = figure_base / 'shap'
data_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
vectorized_path = make_path('data/processed_data/') / dataset_name / 'derived/' / 'fast_text.pkl'
model_path = make_path('data/models/') / dataset_name / 'derived/' / 'fast_text.h5'

# Check for files
# if not fig_dir.exists(): mkdir(fig_dir)
# if not shap_dir.exists(): mkdir(shap_dir)
#
# # Import data
# maps = get_path_maps()
# dir_maps = maps['layer_class']
# name_maps = maps['layer_names']
# layer = 'fast_text'
#
#
# # Get model and feature values
# model = load_model(str(model_path))
# dataset = open_w_pandas(data_path)
# dataset['vectorized_content'] = read_pickle(vectorized_path)
# (train, test), (train_label, test_label) \
#     = split_sets(dataset['vectorized_content'], lambda doc: doc, labels=dataset['is_abusive'])
# train, test, train_label, test_label = to_numpy_array([train, test, train_label, test_label])
#
# explainer = DeepExplainer(model, train)
# shap_values = explainer.shap_values(test)
#
# print(shap_values)
# print(shap_values.shape)

# feature_weights = match_feature_weights(layer_features, weights)
# feature_gains = match_feature_weights(layer_features, gains)
# feature_t_gains = match_feature_weights(layer_features, t_gains)

# Stacked model
# feature_significance(feature_weights, name_maps[layer] + ' Weights',
#                      filename=fig_dir / (layer + '_weight.png'))
# feature_significance(feature_gains, name_maps[layer] + ' Gains', is_weight=False, x_log=True,
#                      filename=fig_dir / (layer + '_gain.png'))
# feature_significance(feature_t_gains, name_maps[layer] + ' Total Gains', is_weight=False, x_log=True,
#                      filename=fig_dir / (layer + '_t_gain.png'))
# shap_feature_significance(model, dataset, name_maps[layer] + ' SHAP Weights', features=layer_features,
#                           filename=shap_dir / (layer + '.png'))

# show()

