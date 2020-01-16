from utilities.data_management import make_path, move_to_root, load_execution_params, check_existence
from pandas import read_csv
from sklearn.decomposition import PCA
from utilities.plotting import scatter_3_plot, show
from numpy import log

move_to_root(4)
max_points = 5000

params = load_execution_params()
dataset = params['dataset']

base_dir = make_path('data/processed_data') / dataset / 'analysis'
embeddings_path = base_dir / 'intent' / 'encoded_documents.csv'
weights_path = base_dir / 'intent_abuse' / 'intent_predictions.csv'

check_existence(embeddings_path)
check_existence(weights_path)

embeddings = read_csv(embeddings_path, header=None).values
weights = read_csv(weights_path, header=None).values
print('Embeddings loaded with shape', embeddings.shape)

if embeddings.shape[1] > 3:
    pca = PCA(whiten=True, random_state=42)
    embeddings = pca.fit_transform(embeddings).transpose()[:3, :max_points]
else:
    embeddings = embeddings.transpose()[:, :max_points]
weights = -log(weights[:max_points, 0])
print('Embeddings ready to plot, data with shape', embeddings.shape, 'and weights with shape', weights.shape)

scatter_3_plot(embeddings, 'plot', weights=weights)
show()
