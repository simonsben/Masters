from utilities.data_management import prepare_csv_reader, prepare_csv_writer


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
    # Get options
    delimiter = options['delimiter'] if 'delimiter' in options else ','
    max_documents = options['max_documents'] if 'max_documents' in options else -1
    encoding = options['encoding'] if 'encoding' in options else None

    # check options
    if type(max_documents) is not int or max_documents < -1:
        raise ValueError('max_documents provided is invalid, give int in range [0, inf]')

    # Opens files
    csv_reader, source_fl, header = prepare_csv_reader(source_filename, delimiter=delimiter, encoding=encoding)
    csv_writer, dest_fl = prepare_csv_writer(dest_filename, save_header)

    for ind, doc in enumerate(csv_reader):  # For each document in source file
        values = [ind]
        content = get_content(doc)

        for process in processes:   # For each pre-processing step to be applied
            value, content = process(content)

            if value is not None:
                values.append(value)

        # Save modified document for destination file
        modified_document = save_content(content, values, doc)
        csv_writer.writerow(modified_document)

        if max_documents != -1 and ind > max_documents:
            break

    # Close files
    source_fl.close()
    dest_fl.close()
