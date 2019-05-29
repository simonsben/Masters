from os import access, W_OK, R_OK, rename
from csv import reader, writer
from pathlib import Path
from pandas import read_csv
from re import search, compile

file_regex = compile(r'\w+\.\w+$')


def make_path(filename):
    """ Makes path from given string """
    return Path(filename) if type(filename) is str else filename


def check_existence(path):
    """ Checks whether the file exists """
    if not path.exists():   # Check if file exists
        raise FileExistsError(path, ' does not exist')


def check_writable(path):
    """ Checks whether the path is valid for writing """
    path = make_path(path)
    is_file = search(file_regex, str(path))
    directory = path.parents[0] if is_file else path

    if not access(directory, W_OK):
        raise PermissionError(directory, ' cannot be written to.')


def check_readable(path):
    """ Checks whether the path is valid for reading """
    path = make_path(path)
    directory = path.parents[0] if path.is_file() else path

    if not access(directory, R_OK):
        raise PermissionError(directory, ' cannot be read from.')


def rename_file(path, new_path):
    """ Renames the given file """
    if not path.is_file(): raise FileNotFoundError('Given path is not for a file')
    if not path.exists(): raise FileExistsError('Given file does not exist')

    rename(path, new_path)


def prepare_csv_reader(file, delimiter=',', has_header=True):
    """ Creates a CSV reader for the specified file """
    path = make_path(file) if type(file) is str else file
    check_existence(path)

    fl = path.open(mode='r')
    csv_reader = reader(fl, delimiter=delimiter)

    header = next(csv_reader) if has_header else None
    return csv_reader, fl, header


def prepare_csv_writer(file, header):
    """ Creates a CSV writer for the specified file """
    path = make_path(file) if type(file) is str else file
    check_writable(path)

    fl = path.open(mode='w', newline='')
    csv_writer = writer(fl, delimiter=',')
    csv_writer.writerow(header)

    return csv_writer, fl


def open_w_pandas(path, columns=None, index_col=False):
    """ Opens file as a Panda dataframe """
    path = make_path(path) if type(path) is str else path
    data_frame = read_csv(path, usecols=columns, index_col=index_col)

    return data_frame
