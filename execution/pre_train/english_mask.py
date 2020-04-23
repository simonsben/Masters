from utilities.data_management import make_path, check_existence, open_w_pandas
from model.analysis import get_english_indexes
from fasttext import load_model
from numpy import savetxt
import config

dataset = config.dataset

model_path = make_path('data/lexicons/fast_text/language_classification.bin')
base_dir = make_path('data/processed_data/') / dataset / 'analysis' / 'intent'
dest_path = base_dir / 'english_mask.csv'

check_existence(model_path)

language_model = load_model(str(model_path))
contexts = open_w_pandas(base_dir / 'contexts.csv')['contexts'].values
print('Loaded model and data.')

mask = get_english_indexes(contexts, language_model, boolean_mask=True)
print('Computed mask.')

savetxt(str(dest_path), mask, fmt='%d', delimiter=',')
