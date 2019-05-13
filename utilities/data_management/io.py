from os import path
from csv import reader


# Check if file exists
def load_csv(filename, delimiter=',', has_header=False, limit_rows=-1):
    if not path.isfile(filename):   # Check if file exists
        raise FileNotFoundError('The given file does not exist')

    with open(filename, 'r') as fl:
        data = []
        csv_reader = reader(fl, delimiter=delimiter)

        if has_header:
            header = next(csv_reader)

        for ind, row_data in enumerate(csv_reader):
            if limit_rows != -1 and ind > limit_rows:
                break

            data.append(row_data)

    if has_header:
        return header, data
    return data
