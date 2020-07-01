from config import dataset, fast_text_model
from pathlib import Path


model_types = {'abuse', 'intent'}
prediction_types = {'abuse', 'intent', 'abusive_intent'}


def get_model_path(model_type, weights=True, index=None):
    """ Generates the path of the model weights """
    if model_type not in model_types:
        raise AttributeError('Supplied model type is invalid.')

    base = Path('data/models') / fast_text_model / 'analysis'

    core_name = model_type + '-' + fast_text_model
    if index is not None:
        core_name += '_' + str(index)

    if weights:
        return base / (core_name + '_weights.h5')
    return base / (core_name + '_model')


def get_embedding_path():
    """ Generates the path to the current FastText model """
    # path = Path('data/models/') / dataset / 'derived' / (dataset + '.bin')
    path = Path('data/lexicons/fast_text/') / (fast_text_model + '.bin')

    if not path.exists():
        raise FileNotFoundError('Embedding model does not exist.')
    return str(path)


def intent_verb_filename(name, model_name):
    """ Generates the filename for intent verb embeddings """
    return name + '_vectors-' + model_name + '.csv.gz'


def get_prediction_path(prediction_type, target=None):
    """ Generates the path for the predictions """
    if prediction_type not in prediction_types:
        raise AttributeError('Supplied model type is invalid.')

    target = target if target is not None else dataset
    base = Path('data/processed_data/') / target / 'analysis' / 'intent_abuse'

    return base / ('%s_predictions.csv' % prediction_type)
