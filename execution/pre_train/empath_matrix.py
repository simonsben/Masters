from empath import Empath
from multiprocessing import Pool
from utilities.data_management import make_path, open_w_pandas, check_existence
from scipy.sparse import csr_matrix, save_npz
import config


def init_function():
    global lexicon
    lexicon = Empath()


def get_empath_val(document):
    return list(lexicon.analyze(document).values())


if __name__ == '__main__':
    dataset_name = config.dataset
    n_threads = config.n_threads

    base_dir = make_path('data/processed_data/') / dataset_name / 'analysis' / 'intent'
    data_path = base_dir / 'contexts.csv'
    dest_path = base_dir / 'context_empaths.npz'

    check_existence(data_path)

    data = open_w_pandas(data_path)
    documents = data['contexts'].values
    print('Data loaded')

    workers = Pool(n_threads, initializer=init_function)

    emp_values = workers.map(get_empath_val, documents)

    workers.close()
    workers.join()

    sparse_values = csr_matrix(emp_values)
    print('sparse rep', sparse_values.nnz)

    save_npz(dest_path, sparse_values)
