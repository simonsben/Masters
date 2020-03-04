from os import listdir, chdir
from pathlib import Path
from scipy.sparse import save_npz
from json import load
from numbers import Number
from csv import field_size_limit
from sys import maxsize

prediction_filename = Path('data/processed_data/predictions.csv')
root_dir_file = '.gitignore'


def move_to_root(max_levels=4, target=root_dir_file):
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


def in_parent_dir():
    """ Checks whether the PWD is the project dir """
    return Path('config.json').is_file()


def load_execution_params(quiet=False):
    """ Loads the current execution parameters from the config file """
    path = Path('config.json')
    with path.open(mode='r') as fl:
        params = load(fl)['execution']

    # Print out key execution parameters
    dataset = params['dataset']
    fast_text_model = params['fast_text_model']

    if not quiet:
        print('Loaded execution params with dataset', dataset, 'and fastText model', fast_text_model)

    return params


def load_dataset_params(dataset=None):
    """ Loads the dataset specific parameters """
    dataset = load_execution_params()['dataset'] if dataset is None else dataset

    tmp = get_path_maps()['dataset_parameters']
    params = {
        param: (tmp[param]
                if isinstance(tmp[param], Number)
                else tmp[param][dataset])
        for param in tmp}

    return params


def make_dir(directory, max_levels=2):
    """ Moves up directory levels until it exists, then moves back down creating child directories """
    if type(directory) is str: directory = Path(directory)
    if directory.suffix != '': directory = directory.parent
    if directory.exists(): return

    levels = [directory] + list(directory.parents)
    for ind, level in enumerate(levels):
        if ind > max_levels: break
        if level.exists():
            for l_ind in range(ind-1, -1, -1):
                levels[l_ind].mkdir()
            return

    raise LookupError('Cannot make dir within', max_levels, 'levels of the specified directory')


def expand_csv_row_size():
    """ Enables the csv reader to accept longer document rows """
    max_size = maxsize
    while True:
        try:
            field_size_limit(max_size)
            break
        except OverflowError:
            max_size = int(max_size / 10)
