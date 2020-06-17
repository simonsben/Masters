from config import dataset, fast_text_model
from pathlib import Path


model_types = {'abuse', 'intent'}


def get_model_path(model_type, weights=True):
    """ Generates the path of the model weights """
    if model_type not in model_types:
        raise AttributeError('Supplied model type is invalid.')

    base = Path('data/models') / dataset / 'analysis'

    core_name = model_type + '-' + fast_text_model
    if weights:
        return base / (core_name + '_weights.h5')
    return base / (core_name + '_model')


def get_embedding_path():
    """ Generates the path to the current FastText model """
    path = Path('data/models/') / dataset / 'derived' / (dataset + '.bin')

    if not path.exists():
        raise FileNotFoundError('Embedding model does not exist.')
    return str(path)


def intent_verb_filename(name, model_name):
    """ Generates the filename for intent verb embeddings """
    return name + '_vectors-' + model_name + '.csv.gz'

