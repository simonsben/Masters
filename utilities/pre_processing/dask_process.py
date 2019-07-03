from utilities.data_management import prepare_csv_reader, prepare_csv_writer
from dask.dataframe import read_csv
from pandas import DataFrame


def is_value_row(process):
    header = process(None, True)
    return header is None, header


def dask_process_documents(source_filename, dest_filename, processes, constants, options):
    """
    Pre-processes all documents within a CSV file.
    Assumed that files are too large to fit in memory, file is processed line-by-line.

    :param source_filename: Filename for the source CSV file
    :param dest_filename: Filename for destination file
    :param processes: List of pre-processing functions, (document_content) -> (value, modified_content)
    :param options: Accepts options for document including
        delimiter of the source file, (default ',')
        max_documents to be pre-processed, (default is entire file)
    """
    # Get options
    delimiter = options['delimiter'] if 'delimiter' in options else ','
    max_documents = options['max_documents'] if 'max_documents' in options else -1
    encoding = options['encoding'] if 'encoding' in options else None
    header = options['header'] if 'header' in options else 'infer'

    # check options
    if type(max_documents) is not int or max_documents < -1:
        raise ValueError('max_documents provided is invalid, give int in range [0, inf]')

    # Opens files
    dataset = read_csv(source_filename, delimiter=delimiter, encoding=encoding, header=header, blocksize=100e6)
    content_label = dataset.columns.values[constants['content']]
    dataset = dataset[[content_label]].rename(columns={content_label: 'content'})

    for process in processes:
        is_none, header = is_value_row(process)
        print(header, is_none)

        def safe_process(doc):
            if type(doc) is str:
                return process(doc)
            return 0, ''

        if is_none:
            def processor(df):
                tmp = df['content'].apply(lambda doc: safe_process(doc))
                # print(tmp)

                df['content'] = [value[1] for value in tmp]
                return df
            dataset = dataset.map_partitions(processor).persist()
        else:
            def processor(df):
                tmp = df['content'].apply(lambda doc: safe_process(doc))
                # print(tmp)

                df['content'] = [value[1] for value in tmp]
                df[header] = [value[0] for value in tmp]
                return df
            dataset = dataset.map_partitions(processor).persist()

    print('Execution tree built, executing and saving')
    print(dataset)
    dataset.to_csv(dest_filename, compute=True)
