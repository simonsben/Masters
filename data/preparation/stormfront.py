from utilities.data_management import make_path, check_readable, rename_file, prepare_csv_writer, prepare_csv_reader, \
    expand_csv_row_size, load_execution_params, move_to_root
from re import compile
from unidecode import unidecode
from multiprocessing import Pool
from functools import partial

newline_regex = compile(r'(<br />)[\n\r]?n|[\n\r]n')


def apply_processing(line):
    line = line[3:]
    line[2] = newline_regex.sub(' ', unidecode(line[2]))

    return line


if __name__ == '__main__':
    move_to_root()
    expand_csv_row_size()
    n_threads = load_execution_params()['n_threads']

    # Generate and check path
    directory_filename = 'data/datasets/storm-front/'
    filename = 'storm-front'
    directory = make_path(directory_filename)
    source_path = directory / (filename + '_backup.csv')
    dest_path = directory / (filename + '.csv')

    file_header = ['date', 'user', 'document_content']

    # Backup file
    if not source_path.exists():
        check_readable(dest_path)
        rename_file(dest_path, source_path)

    # Initialize csv reader and writer
    csv_writer, dest_file = prepare_csv_writer(dest_path, file_header)
    csv_reader, source_file, _ = prepare_csv_reader(source_path, encoding='utf-8', has_header=False)
    print('Csv reader/writer ready, processing')

    workers = Pool(n_threads)
    dataset = list(workers.imap(apply_processing, csv_reader))
    workers.close()
    workers.join()

    # Prepare rows
    for line in dataset:
        csv_writer.writerow(line)

    source_file.close()
    dest_file.close()
