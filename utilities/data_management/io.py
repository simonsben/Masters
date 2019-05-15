from os import access, W_OK, R_OK
from csv import reader, writer
from pathlib import Path
from pandas import read_csv


def make_path(filename):
    """ Makes path from given string """
    return Path(filename) if type(filename) is str else filename


def check_existance(path):
    """ Checks whether the file exists """
    if not path.exists():   # Check if file exists
        raise FileExistsError('The given file does not exist')


def check_writable(path):
    """ Checks whether the path is valid for writing """
    path = make_path(path)
    directory = path.parents[0] if path.is_file() else path

    if not access(directory, W_OK):
        raise PermissionError('The given file location cannot be written to.')


def check_readable(path):
    """ Checks whether the path is valid for reading """
    path = make_path(path)
    directory = path.parents[0] if path.is_file() else path

    if not access(directory, R_OK):
        raise PermissionError('The given file location cannot be read from.')


def prepare_csv_reader(file, delimiter=',', has_header=True):
    """ Creates a CSV reader for the specified file """
    path = make_path(file) if type(file) is str else file
    check_existance(path)

    fl = path.open(mode='r')
    csv_reader = reader(fl, delimiter=delimiter)

    header = next(csv_reader) if has_header else None
    return csv_reader, fl, header


def prepare_csv_writer(file, header):
    """ Creates a CSV writer for the specified file """
    path = make_path(file) if type(file) is str else file
    check_writable(path)

    fl = path.open(mode='w', newline='')
    csv_writer = writer(fl, delimiter=',', quotechar='"')
    csv_writer.writerow(header)

    return csv_writer, fl


def open_w_pandas(path, columns=None):
    """ Opens file as a Panda dataframe """
    path = make_path(path) if type(path) is str else path
    data_frame = read_csv(path, usecols=columns)

    return data_frame
