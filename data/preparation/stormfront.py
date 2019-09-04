from utilities.data_management import make_path, check_readable, rename_file, expand_csv_row_size, \
    load_execution_params, move_to_root
from re import compile
from unidecode import unidecode
from multiprocessing import Pool
from pandas import read_csv, DataFrame
from numpy import ndarray

newline_regex = compile(r'(<br />)[\n\r]?n|[\n\r]n')
full_run = True


def apply_processing(line):
    if not isinstance(line, ndarray) or len(line) < 3:
        print('blablabla')
        return None
    line = line[3:]
    line[2] = newline_regex.sub(' ', unidecode(line[2])) if isinstance(line[2], str) else ''

    return line


if __name__ == '__main__':
    move_to_root()
    expand_csv_row_size()
    n_threads = load_execution_params()['n_threads']

    # Generate and check path
    filename = 'storm-front' + ('-full' if full_run else '')
    directory = make_path('data/datasets/') / filename
    source_path = directory / (filename + '_backup.csv')
    dest_path = directory / (filename + '.csv')

    file_header = ['date', 'user', 'document_content']

    # Backup file
    if not source_path.exists():
        check_readable(dest_path)
        rename_file(dest_path, source_path)

    # Initialize csv reader and writer
    dataset = read_csv(source_path, encoding='utf-8').values
    print('File loaded')

    workers = Pool(n_threads)
    dataset = workers.map(apply_processing, dataset)
    workers.close()
    workers.join()

    dataset = DataFrame(dataset, columns=file_header)
    dataset.to_csv(dest_path)
