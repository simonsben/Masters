from os import access, W_OK, R_OK, rename
from csv import reader, writer, QUOTE_NONNUMERIC
from pathlib import Path
from pandas import read_csv, DataFrame
from re import search, compile
from dask.dataframe import read_csv as dask_read
from numpy import asarray, savetxt, object
from sys import argv

file_regex = compile(r'\w+\.\w+$')


def make_path(filename):
    """ Makes path from given string """
    return Path(filename) if type(filename) is str else filename


def check_existence(paths):
    """ Checks whether the file exists """
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if isinstance(path, str):
            path = Path(path)
        if not isinstance(path, Path):
            raise TypeError('Provided path', path, 'is not of type Path.')
        if not path.exists():   # Check if file exists
            raise FileExistsError(path, 'does not exist.')


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


def open_w_pandas(path, columns=None, index_col=0, encoding=None):
    """ Opens file as a Panda dataframe """
    path = make_path(path)
    data_frame = read_csv(path, usecols=columns, index_col=index_col, encoding=encoding)

    return data_frame


def save_dataframe(data_frame, path):
    """ Saves DataFrame with standard parameters """
    path = make_path(path)
    data_frame.to_csv(path, quoting=QUOTE_NONNUMERIC)


def open_w_dask(path, index_col=0, dtypes=None):
    path = make_path(path)
    data_frame = dask_read(path, dtype=dtypes)

    if index_col is None:
        return data_frame
    indexes = list(range(data_frame.shape[1]))
    indexes.remove(index_col)
    return data_frame.iloc[:, indexes]


def open_embeddings(path):
    """ Loads pre-calculated embeddings """
    path = make_path(path)
    raw_embeddings = read_csv(path)

    tokens = raw_embeddings.values[:, 0]
    embeddings = raw_embeddings.values[:, 1:].astype(float)

    return tokens, embeddings


def open_exp_lexicon(path, raw=False):
    """ Opens an expanded lexicon """
    csv_reader, fl, header = prepare_csv_reader(path)

    if raw:
        return [header] + [expanded_terms for expanded_terms in csv_reader]

    terms = set(header)
    for new_terms in csv_reader:
        terms = terms.union(new_terms)

    return DataFrame(terms, columns=['word'])


def convert_to_parquet(path, data=None):
    """ Converts a csv file to a parquet file """
    if data is None:
        data = open_w_pandas(path.parent / (path.stem + '.csv'))
    else:
        if not isinstance(data, DataFrame):
            raise TypeError('data must be a pandas DataFrame, given ' + str(type(data)))

    if path.suffix == '':
        raise ValueError('Must provide a filepath, not a directory')

    filename = path.parent / (path.stem + '.parquet')
    data.to_parquet(filename, compression='gzip')


def load_tsv(path, has_header=False):
    """ Opens a tsv file """
    csv_reader, fl, header = prepare_csv_reader(path, delimiter='\t', has_header=has_header)

    data = [header] if has_header else []
    for line in csv_reader:
        data.append(line)

    return data


def write_context_map(filename, context_map):
    file_path = make_path(filename)

    with file_path.open('w') as fl:
        for key in context_map:
            tmp = context_map[key]
            line = [str(val) for val in [key, tmp.start, tmp.stop - 1]]
            fl.write(','.join(line) + '\n')


def output_abusive_intent(index_set, predictions, contexts, filename=None):
    """ Prints abusive intent results to console and saves to disk """
    index_set = asarray(list(index_set))
    abuse, intent, abusive_intent = predictions

    if filename is not None:
        savetxt(filename, index_set, delimiter=',', fmt='%d')

    print('%10s %8s %8s %8s  %s' % ('index', 'hybrid', 'intent', 'abuse', 'context'))
    for index in index_set:
        print('%10d %8.4f %8.4f %8.4f  %s' % (index, abusive_intent[index], intent[index], abuse[index], contexts[index]))


type_map = {
    'O': '%s',
    'i': '%d',
    'f': '%.6f'
}


def vector_to_file(data_vector, filename):
    """ Saves a numpy vector to a csv without a column header """
    data_type = data_vector.dtype.kind
    if data_type not in type_map:
        raise TypeError('Unsupported type,', data_type)

    savetxt(filename, data_vector, delimiter=',', fmt=type_map[data_type])


def load_vector(file_path):
    """ Loads data from a csv with a single column and no header """
    file_path = make_path(file_path)

    return read_csv(file_path, header=None)[0].values


def check_execution_targets():
    """ Checks if the python arguments specify valid file paths for consumption """
    targets = [Path(target) for target in argv[1:]] if len(argv) > 1 else [None]

    for target in targets:
        if target is None or not target.exists():
            print('Specified data does not exist, using environment target.')
            return False
    return True
