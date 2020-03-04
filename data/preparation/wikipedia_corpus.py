from utilities.data_management import make_path, move_to_root, load_execution_params
from os import rename
from multiprocessing import Pool
from itertools import compress
from pandas import DataFrame


def long_enough(document, threshold=10):
    n_words = len(document.split(' '))

    return n_words >= threshold


if __name__ == '__main__':
    move_to_root()
    directory = make_path('data/datasets/wikipedia_corpus')
    file_path = directory / 'wikipedia_corpus.csv'
    backup_path = directory / 'wikipedia_corpus_backup.csv'

    if not (file_path.exists() or backup_path.exists()):
        raise FileNotFoundError('File doesn\'t exist.')
    if not backup_path.exists():
        print('Backing up.')
        rename(file_path, backup_path)
    print('Config done.')

    with backup_path.open(encoding='utf-8') as fl:
        content = fl.readlines()
    print('Content loaded')

    n_thread = load_execution_params()['n_threads']
    pool = Pool(n_thread)

    non_header_mask = pool.map(long_enough, content)

    pool.close()
    pool.join()
    print('Computed mask')

    content = list(compress(content, non_header_mask))
    content = DataFrame(content, columns=['document_content'])
    print('Converted to DataFrame.')

    content.to_csv(file_path, encoding='utf-8')

    # index = 0
    # with file_path.open('w', encoding='utf-8') as fl:
    #     for document in compress(content, non_header_mask):
    #         fl.write(str(index) + ',' + document)
    #         index += 1

    print('Done.')
