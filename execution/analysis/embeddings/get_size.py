from utilities.data_management import make_path, check_existence
from config import fast_text_model
from fasttext import load_model

embedding_path = make_path('data/lexicons/fast_text') / (fast_text_model + '.bin')

check_existence(embedding_path)

model = load_model(embedding_path.__str__())

print('Model has %d dimensions' % model.get_dimension())
