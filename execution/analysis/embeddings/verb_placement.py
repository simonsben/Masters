from utilities.data_management import move_to_root, make_path, check_existence, load_execution_params, open_w_pandas
from model.analysis import intent_verb_filename, cluster_verbs
from scipy.linalg import svd
from utilities.analysis import normalize_embeddings
from utilities.plotting import scatter_plot, show, scatter_3_plot, plot_dendrogram
from sklearn.decomposition import PCA


move_to_root(4)

params = load_execution_params()
dataset = params['dataset']
model_name = params['fast_text_model']

base_dir = make_path('data/processed_data') / dataset / 'analysis' / 'embeddings'
action_path = base_dir / intent_verb_filename('action', model_name)
desire_path = base_dir / intent_verb_filename('desire', model_name)

check_existence(action_path)
check_existence(desire_path)
print('Completed config.')

action = open_w_pandas(action_path).values
desire = open_w_pandas(desire_path).values
print('Loaded data.')

action_tokens = action[:, 0]
action_vectors = normalize_embeddings(action[:, 1:].astype(float))

desire_tokens = desire[:, 0]
desire_vectors = normalize_embeddings(desire[:, 1:].astype(float))

pca = PCA(random_state=420)
reduced_action = pca.fit_transform(action_vectors)
reduced_desire = pca.fit_transform(desire_vectors)
print('Normalized verbs.')

# scatter_plot(action_weights, 'Action singular values')
# scatter_plot(desire_weights, 'Desire singular values')

action_model = cluster_verbs(reduced_action)
desire_model = cluster_verbs(reduced_desire)
print('Completed clustering')

plot_dendrogram(action_model, action_tokens[:action_model.n_leaves_], 'Dendrogram of action verbs')
plot_dendrogram(desire_model, desire_tokens[:desire_model.n_leaves_], 'Dendrogram of desire verbs')

show()
