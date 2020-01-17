from utilities.data_management import make_path, move_to_root, load_execution_params, check_existence
from pandas import read_csv
from sklearn.decomposition import PCA
from utilities.plotting import scatter_3_plot, show
from numpy import log, any, where

move_to_root(4)
max_points = 5000

params = load_execution_params()
dataset = params['dataset']

base_dir = make_path('data/processed_data') / dataset / 'analysis'
embeddings_path = base_dir / 'intent' / 'encoded_documents.csv'
weights_path = base_dir / 'intent_abuse' / 'intent_predictions.csv'
figure_dir = make_path('figures') / dataset / 'analysis' / 'intent' / 'document_embeddings.png'

check_existence(embeddings_path)
check_existence(weights_path)

embeddings = read_csv(embeddings_path, header=None).values
weights = read_csv(weights_path, header=None).values
print('Embeddings loaded with shape', embeddings.shape)

threshold = .95

plot_selection = where(
    any([
        weights < (1 - threshold),
        weights > threshold
    ], axis=0).reshape(-1)
    )[0][:max_points]

print('plot selection with shape', plot_selection.shape)

if embeddings.shape[1] > 3:
    pca = PCA(whiten=True, random_state=42)
    embeddings = pca.fit_transform(embeddings)[plot_selection].transpose()[:3]
else:
    embeddings = embeddings.transpose()[:, plot_selection]
weights = weights[plot_selection, 0]
print('Embeddings ready to plot, data with shape', embeddings.shape, 'and weights with shape', weights.shape)

axis_titles = ['Auto-encoder axis ' + str(i) for i in range(1, 4)]

scatter_3_plot(embeddings, 'Document LSTM auto-encoder embeddings', weights=weights, c_bar_title='Predicted intent',
               ax_titles=axis_titles)
show()
