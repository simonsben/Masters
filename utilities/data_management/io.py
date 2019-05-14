from os import access, W_OK
from csv import reader, writer
from pathlib import Path


def make_path(filename):
    """ Makes path from given string """
    return Path(filename)


def check_existance(path):
    """ Checks whether the file exists """
    if not path.exists():   # Check if file exists
        raise FileExistsError('The given file does not exist')


# Function to check if the path is valid
def check_writable(path):
    """ Checks whether the path is valid for writing """
    directory = path.parents[0]

    if not access(directory, W_OK):
        raise PermissionError('The given file location cannot be written to')


# Checks filename and open file with a csv reader
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


# TODO remove - old function
def load_csv(filename, delimiter=',', has_header=False, limit_rows=-1):
    """ Loads contents of a CSV file """
    csv_reader, fl, header = prepare_csv_reader(filename, delimiter, has_header)

    data = []
    for ind, row_data in enumerate(csv_reader):
        if limit_rows != -1 and ind > limit_rows:
            break

        data.append(row_data)
    fl.close()

    if has_header:
        return header, data
    return data
