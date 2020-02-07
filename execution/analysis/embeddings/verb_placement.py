from utilities.data_management import move_to_root, make_path, check_existence, load_execution_params, open_w_pandas
from model.analysis import intent_verb_filename
from scipy.linalg import svd
from utilities.analysis import normalize_embeddings
from utilities.plotting import scatter_plot, show, scatter_3_plot


move_to_root(4)

params = load_execution_params()
dataset = params['dataset']
model_name = params['fast_text_model']

base_dir = make_path('data/processed_data') / dataset / 'analysis' / 'embeddings'
action_path = base_dir / intent_verb_filename('action', model_name)
desire_path = base_dir / intent_verb_filename('desire', model_name)

check_existence(action_path)
check_existence(desire_path)

action = open_w_pandas(action_path).values
desire = open_w_pandas(desire_path).values

action_tokens = action[:, 0]
action_vectors = normalize_embeddings(action[:, 1:].astype(float))

desire_tokens = desire[:, 0]
desire_vectors = normalize_embeddings(desire[:, 1:].astype(float))

reduced_action, action_weights, _ = svd(action_vectors)
reduced_desire, desire_weights,  _ = svd(desire_vectors)

scatter_plot(action_weights, 'Action singular values')
scatter_plot(desire_weights, 'Desire singular values')

scatter_3_plot(reduced_action[:, :3].T, 'Action SVD')

show()
