from utilities.data_management import load_execution_params
from multiprocessing import Pool
from pandas import DataFrame, read_csv
from functools import partial


def apply_process(packed_data, processes, get_content, save_content):
    """
    Applies the pre-processing filters to a document
    :param packed_data: a tuple of document index, document
    :param processes: list of processes to be applied
    :param get_content: function that acts as an accessor for the dataset
    :param save_content: function that acts a mutator (ish..) for the dataset
    :return: list of pre-processing values and
    """
    index, document = packed_data

    values = [index]
    content = get_content(document)

    for process in processes:   # For each pre-processing step to be applied
        value, content = process(content if isinstance(content, str) else '')

        if value is not None:
            values.append(value)

    return save_content(content, values, document)


def process_documents(source_filename, dest_filename, processes, get_content, save_content, save_header, options):
    """
    Pre-processes all documents within a CSV file.
    Assumed that files are too large to fit in memory, file is processed line-by-line.

    :param source_filename: Filename for the source CSV file
    :param dest_filename: Filename for destination file
    :param processes: List of pre-processing functions, (document_content) -> (value, modified_content)
    :param get_content: Accessor for source file, (document) -> (document_content)
    :param save_content: Mutator for destination file (row) (modified_content, values, document) -> (modified_document)
    :param save_header: Header for destination file, List
    :param options: Accepts options for document including
        delimiter of the source file, (default ',')
        max_documents to be pre-processed, (default is entire file)
    """
    delimiter = options['delimiter'] if 'delimiter' in options else ','
    max_documents = options['max_documents'] if 'max_documents' in options else -1
    encoding = options['encoding'] if 'encoding' in options else None

    n_threads = load_execution_params()['n_threads']

    dataset = read_csv(source_filename, delimiter=delimiter, encoding=encoding, index_col=0).values
    if max_documents is not None:
        dataset = dataset[:max_documents]

    workers = Pool(n_threads)
    processed_data = workers.map(
        partial(apply_process, processes=processes, get_content=get_content, save_content=save_content),
        enumerate(dataset)
    )
    workers.close()
    workers.join()

    processed_data = DataFrame(processed_data, columns=save_header)
    processed_data.to_csv(dest_filename, index=False)
