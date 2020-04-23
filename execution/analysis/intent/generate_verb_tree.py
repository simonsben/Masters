from utilities.data_management import move_to_root, make_path, load_vector, open_w_pandas, check_existence
from model.analysis import intent_verb_filename
from model.expansion.verb_tree import build_tree_and_collect_leaves
from utilities.plotting import plot_dendrogram, show
import config

target_action_labels = ['kill', 'fight', 'eliminate']
target_desire_labels = ['wish', 'hope', 'try']

move_to_root()
model_name = config.fast_text_model
dataset = config.dataset

base = make_path('data/processed_data/') / dataset / 'analysis'
embedding_dir = base / 'embeddings'
intent_dir = base / 'intent'

action_path = embedding_dir / intent_verb_filename('action', model_name)
desire_path = embedding_dir / intent_verb_filename('desire', model_name)
figure_dir = make_path('figures') / dataset / 'analysis'

check_existence(action_path)
check_existence(desire_path)
print('Completed config.')

action = open_w_pandas(action_path).values
desire = open_w_pandas(desire_path).values
print('Loaded data.')

initial_mask = load_vector(intent_dir / 'intent_mask.csv')
contexts = open_w_pandas(intent_dir / 'contexts.csv')['contexts'].values
print('Content loaded.')

action_model, action_leaves, action_tokens = build_tree_and_collect_leaves(action, target_action_labels)
print(action_leaves)

desire_model, desire_leaves, desire_tokens = build_tree_and_collect_leaves(desire, target_desire_labels)
print(desire_leaves)

fig_size = (15, 7)
plot_dendrogram(action_model, action_tokens, 'Action dendrogram', figsize=fig_size)
plot_dendrogram(desire_model, desire_tokens, 'Desire dendrogram', figsize=fig_size)

show()
