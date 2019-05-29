from os import listdir, chdir
from pathlib import Path
from scipy.sparse import save_npz
from json import load

prediction_filename = Path('data/processed_data/predictions.csv')
target_file = '.gitignore'


def move_to_root(max_levels=3, target=target_file):
    """
     Changes PWD to root project directory.
     NOTE: By default assumes there is only one .gitignore file in project files.
    :param max_levels: Maximum number of directory levels checked, (default 3)
    :param target: Target file, must only be one in project files, (default .gitignore)
    """
    for i in range(max_levels):
        files = listdir('.')
        if target in files:
            return

        chdir('../')
    raise FileNotFoundError('Could not find .gitignore file, is the current dir more than', max_levels, 'deep?')


def save_prepared(directory, model_name, train, test):
    """ Save prepared sparse matrix """
    save_npz(directory / (model_name + '_train.npz'), train)
    save_npz(directory / (model_name + '_test.npz'), test)


def get_path_maps():
    """ Loads the path maps from data/accessors/constants.json """
    path = Path('data/accessors/constants.json')
    with path.open(mode='r') as fl:
        maps = load(fl)
    return maps
