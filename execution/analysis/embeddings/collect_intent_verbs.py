from fasttext import load_model
from utilities.data_management import make_path, check_existence, make_dir, intent_verb_filename, load_vector, \
    get_embedding_path
from pandas import read_csv
from model.analysis import get_verbs, generate_word_vectors
from config import dataset, fast_text_model


model_path = get_embedding_path()
base_dir = make_path('data/processed_data/') / dataset / 'analysis'
data_dir = base_dir / 'intent'
destination_dir = base_dir / 'embeddings'

frame_info_path = data_dir / 'intent_frame.csv'
english_mask = data_dir / 'english_mask.csv'

desire_index = 1
action_index = 2

check_existence([frame_info_path, model_path, english_mask])
make_dir(destination_dir)
print('Config complete.')

english_mask = load_vector(english_mask).astype(bool)
intent_frames = read_csv(frame_info_path, header=None, keep_default_na=False).values[english_mask]
print('Loaded data with shape', intent_frames.shape)

desire_verbs = get_verbs(intent_frames, desire_index)
action_verbs = get_verbs(intent_frames, action_index)
print('Isolated verbs.')

model = load_model(model_path)
print('Loaded model.')

desire_vectors = generate_word_vectors(desire_verbs, model)
desire_vectors.to_csv(destination_dir / intent_verb_filename('desire', fast_text_model), compression='gzip')
print('Completed desire verbs.')

action_vectors = generate_word_vectors(action_verbs, model)
action_vectors.to_csv(destination_dir / intent_verb_filename('action', fast_text_model), compression='gzip')
print('Completed action verbs.')
