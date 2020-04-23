from fasttext import load_model
from utilities.data_management import move_to_root, make_path, check_existence, make_dir
from pandas import read_csv
from model.analysis import get_verbs, intent_verb_filename, generate_word_vectors
import config

move_to_root()

dataset = config.dataset
model_name = config.fast_text_model

model_path = make_path('data/lexicons/fast_text/') / (model_name + '.bin')
base_dir = make_path('data/processed_data/') / dataset / 'analysis'
data_dir = base_dir / 'intent'
destination_dir = base_dir / 'embeddings'

frame_info_path = data_dir / 'intent_frame.csv'
english_mask = data_dir / 'english_mask.csv'

desire_index = 1
action_index = 2

make_dir(destination_dir)
check_existence(frame_info_path)
check_existence(model_path)
check_existence(english_mask)
print('Config complete.')

english_mask = read_csv(english_mask, header=None)[0].values.astype(bool)
intent_frames = read_csv(frame_info_path, header=None, keep_default_na=False).values[english_mask]
print('Loaded data with shape', intent_frames.shape)

desire_verbs = get_verbs(intent_frames, desire_index)
action_verbs = get_verbs(intent_frames, action_index)
print('Isolated verbs.')

model = load_model(str(model_path))
print('Loaded model.')

desire_vectors = generate_word_vectors(desire_verbs, model)
desire_vectors.to_csv(destination_dir / intent_verb_filename('desire', model_name), compression='gzip')
print('Completed desire verbs.')

action_vectors = generate_word_vectors(action_verbs, model)
action_vectors.to_csv(destination_dir / intent_verb_filename('action', model_name), compression='gzip')
print('Completed action verbs.')
