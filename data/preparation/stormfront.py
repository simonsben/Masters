from utilities.data_management import make_path, check_readable, rename_file, expand_csv_row_size, \
    load_execution_params, move_to_root
from re import compile
from unidecode import unidecode
from multiprocessing import Pool
from pandas import read_csv, DataFrame, to_datetime, concat
from numpy import ndarray

newline_regex = compile(r'(<br />)[\n\r]?n|[\n\r]n')
full_run = True

extension_filename = 'storm-front-extension.tsv'
file_header = ['date', 'user', 'document_content']
dataset_columns = [3, 4, 5]


def apply_processing(line):
    if not isinstance(line, ndarray) or len(line) < 3:
        print('Bad line', line)
        return None
    line = line[dataset_columns]
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
    extension_path = directory / extension_filename
    dest_path = directory / (filename + '.csv')

    # Backup file
    if not source_path.exists():
        check_readable(dest_path)
        rename_file(dest_path, source_path)
    check_readable(extension_path)

    # Initialize csv reader and writer
    dataset = read_csv(source_path, encoding='utf-8', header=None).values
    dataset_extension = read_csv(extension_path, delimiter='\t', error_bad_lines=False).values
    print('File loaded, applying processing')

    workers = Pool(n_threads)

    dataset = workers.map(apply_processing, dataset)
    dataset_extension = workers.map(apply_processing, dataset_extension)

    workers.close()
    workers.join()
    print('Applied processing, converting to dataframe')

    dataset = DataFrame(dataset, columns=file_header)
    dataset_extension = DataFrame(dataset_extension, columns=file_header)
    print('Converted to dataframe, cleaning dates')

    dataset['date'] = to_datetime(dataset['date'], errors='coerce')
    dataset_extension['date'] = to_datetime(dataset_extension['date'], errors='coerce')
    print('Cleaned dates, extending dataset')

    dataset = concat([dataset, dataset_extension])
    print('Dataset extended, saving')

    dataset.to_csv(dest_path)
