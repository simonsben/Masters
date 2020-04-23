from utilities.data_management import make_path, load_vector, open_w_pandas, check_existence, read_csv
from model.analysis import intent_verb_filename, refine_mask
from numpy import asarray, logical_not, all, sum, savetxt
from utilities.plotting import plot_dendrogram, show, hist_plot
from model.expansion.verb_tree import build_tree_and_collect_leaves, get_sub_tree
from model.expansion.verb_space import get_cube_mask, get_cone_mask
import config

target_action_labels = ['kill', 'fight', 'act', 'take']
target_desire_labels = ['want', 'need', 'going', 'have', 'about', 'planning']

model_name = config.fast_text_model
dataset = config.dataset

base = make_path('data/processed_data/') / dataset / 'analysis'
embedding_dir = base / 'embeddings'
intent_dir = base / 'intent'

action_path = embedding_dir / intent_verb_filename('action', model_name)
desire_path = embedding_dir / intent_verb_filename('desire', model_name)
verb_path_generator = lambda name: embedding_dir / (name + '_verbs.csv')
figure_dir = make_path('figures') / dataset / 'analysis'

check_existence(action_path)
check_existence(desire_path)
print('Completed config.')

action = open_w_pandas(action_path).values
desire = open_w_pandas(desire_path).values
print('Loaded data.')

initial_mask = load_vector(intent_dir / 'intent_mask.csv')
contexts = open_w_pandas(intent_dir / 'contexts.csv')['contexts'].values
intent_frames = read_csv(intent_dir / 'intent_frame.csv', header=None).values
print('Content loaded.')

max_verbs = None
desire_verb_index = 1
action_verb_index = 2

print('Tree')
action_model, action_leaves, action_tokens, action_vectors = build_tree_and_collect_leaves(action, target_action_labels, max_labels=max_verbs)
print(action_leaves)

desire_model, desire_leaves, desire_tokens, desire_vectors = build_tree_and_collect_leaves(desire, target_desire_labels, max_labels=max_verbs)
print(desire_leaves)

refined_tree_desire_mask = refine_mask(initial_mask, desire_leaves, intent_frames, desire_verb_index)
print(1 - sum(refined_tree_desire_mask != initial_mask) / sum(initial_mask == 1))


print('Cube')
cube_action_tokens, cube_action_mask = get_cube_mask(action, target_action_labels)
print(cube_action_tokens)

cube_desire_tokens, cube_desire_mask = get_cube_mask(desire, target_desire_labels)
print(cube_desire_tokens)

refined_cube_desire_mask = refine_mask(initial_mask, cube_desire_tokens, intent_frames, desire_verb_index)
print(1 - sum(refined_cube_desire_mask != initial_mask) / sum(initial_mask == 1))


print('Cone')
cone_action_tokens, cone_action_mask, _ = get_cone_mask(action, target_action_labels)
print(cone_action_tokens)

cone_desire_tokens, cone_desire_mask, distances = get_cone_mask(desire, target_desire_labels)
print(cone_desire_tokens)

hist_plot(distances, 'Histogram of distances to central desire vector')

refined_cone_desire_mask = refine_mask(initial_mask, cone_desire_tokens, intent_frames, desire_verb_index)
print(1 - sum(refined_cone_desire_mask != initial_mask) / sum(initial_mask == 1))

savetxt(intent_dir / 'tree_mask.csv', refined_tree_desire_mask, fmt='%.1f')
savetxt(intent_dir / 'cube_mask.csv', refined_cube_desire_mask, fmt='%.1f')
savetxt(intent_dir / 'cone_mask.csv', refined_cone_desire_mask, fmt='%.1f')

# sub_dimensions = 100
# sub_action_model, sub_action_mask = get_sub_tree(action_leaves, action_tokens, action_vectors)
# sub_desire_model, sub_desire_mask = get_sub_tree(desire_leaves, desire_tokens, desire_vectors)
#
# plot_dendrogram(sub_action_model, action_tokens[sub_action_mask][:sub_dimensions], 'Action sub-tree dendrogram', figsize=(15, 8))
# plot_dendrogram(sub_desire_model, desire_tokens[sub_desire_mask][:sub_dimensions], 'Desire sub-tree dendrogram', figsize=(15, 8))
#
# show()
