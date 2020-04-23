from numpy import sum, divide, cumsum
from matplotlib.pyplot import show, savefig
from multiprocessing import Pool
from functools import partial
from time import time


def document_term_rate(document, index):
    unique_terms = set()
    term_rate = [0]

    for term in document[index].split(' '):
        term_value = 0
        if term not in unique_terms:
            unique_terms.add(term)
            term_value = 1

        term_rate.append(term_rate[-1] + term_value)

    term_rate.pop(0)
    return term_rate


def fold_in_list(a, b):
    large, small = (a, b) if len(a) > len(b) else (b, a)
    small_len = len(small)

    return list(sum([small, large[:small_len]], axis=0)) + large[small_len:]


def corpus_term_rate(documents, index):
    # work_pool = Pool(n_thread)
    # term_rates = work_pool.map(partial(document_term_rate, index=index), documents)
    # work_pool.close()
    # work_pool.join()

    corpus_rate = []
    num_docs = 0
    for document in documents:
        rate = document_term_rate(document, index)
        corpus_rate = fold_in_list(corpus_rate, rate)
        num_docs += 1

        if num_docs % 500000 == 0:
            print(num_docs)

    return cumsum(divide(corpus_rate, num_docs))
    # return divide(corpus_rate, len(term_rates))


if __name__ == '__main__':
    from utilities.data_management import make_path, move_to_root, check_existence, prepare_csv_reader, \
        expand_csv_row_size
    from utilities.plotting import scatter_plot
    import config

    move_to_root(4)
    expand_csv_row_size()

    n_thread = config.n_threads
    dataset_name = params['dataset']
    dataset_path = make_path('data/prepared_data/') / (dataset_name + '.csv')
    fig_path = make_path('figures') / dataset_name / 'analysis' / 'term_rate.png'

    check_existence(dataset_path)

    reader, fl, header = prepare_csv_reader(dataset_path)
    content_ind = header.index('document_content')

    start = time()
    rate = corpus_term_rate(reader, content_ind)
    print('Done in', time() - start)

    x = list(range(1, len(rate) + 1))
    ax = scatter_plot((x, rate), 'New term rate in corpus', size=3)

    ax.set_xlabel('Term in document')
    ax.set_ylabel('Average number of unique terms')

    savefig(fig_path)

    show()
