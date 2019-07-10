from os import access, W_OK, R_OK, rename
from csv import reader, writer, QUOTE_NONE
from pathlib import Path
from pandas import read_csv, DataFrame
from re import search, compile
from dask.dataframe import read_csv as dask_read

file_regex = compile(r'\w+\.\w+$')


def make_path(filename):
    """ Makes path from given string """
    return Path(filename) if type(filename) is str else filename


def check_existence(path):
    """ Checks whether the file exists """
    if not path.exists():   # Check if file exists
        raise FileExistsError(path, 'does not exist')


def check_writable(path):
    """ Checks whether the path is valid for writing """
    path = make_path(path)
    is_file = search(file_regex, str(path))
    directory = path.parents[0] if is_file else path

    if not access(directory, W_OK):
        raise PermissionError(directory, 'cannot be written to.')


def check_readable(path):
    """ Checks whether the path is valid for reading """
    path = make_path(path)
    directory = path.parents[0] if path.is_file() else path

    if not access(directory, R_OK):
        raise PermissionError(directory, 'cannot be read from.')


def rename_file(path, new_path):
    """ Renames the given file """
    if not path.is_file(): raise FileNotFoundError('Given path is not for a file')
    if not path.exists(): raise FileExistsError('Given file does not exist')

    rename(path, new_path)


def prepare_csv_reader(file, delimiter=',', has_header=True, encoding=None):
    """ Creates a CSV reader for the specified file """
    path = make_path(file)
    check_existence(path)

    fl = path.open(mode='r', encoding=encoding)
    csv_reader = reader(fl, delimiter=delimiter)

    header = next(csv_reader) if has_header else None
    return csv_reader, fl, header


def prepare_csv_writer(file, header=None):
    """ Creates a CSV writer for the specified file """
    path = make_path(file)
    check_writable(path)

    fl = path.open(mode='w', newline='')
    csv_writer = writer(fl, delimiter=',')

    if header is not None:
        csv_writer.writerow(header)

    return csv_writer, fl


def open_w_pandas(path, columns=None, index_col=0):
    """ Opens file as a Panda dataframe """
    path = make_path(path)
    data_frame = read_csv(path, usecols=columns, index_col=index_col)

    return data_frame


def open_w_dask(path, index_col=0, dtypes=None):
    path = make_path(path)
    data_frame = dask_read(path, dtype=dtypes)

    if index_col is None:
        return data_frame
    indexes = list(range(data_frame.shape[1]))
    indexes.remove(index_col)
    return data_frame.iloc[:, indexes]


def open_fast_embed(path):
    """ Opens a FastText embedding file (.vec) """
    path = make_path(path)
    embedding = read_csv(path, quoting=QUOTE_NONE, delimiter=' ', skiprows=1, header=None)

    return embedding


def open_exp_lexicon(path, raw=False):
    """ Opens an expanded lexicon """
    csv_reader, fl, header = prepare_csv_reader(path)

    if raw:
        return [header] + [expanded_terms for expanded_terms in csv_reader]

    terms = set(header)
    for new_terms in csv_reader:
        terms = terms.union(new_terms)

    return DataFrame(terms, columns=['word'])
