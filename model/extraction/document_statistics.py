from pandas import DataFrame, SparseDataFrame, notnull
from scipy.sparse import csr_matrix


def document_statistics(dataset):
    """
    Takes pre-processing and document statistics then constructs a document-stat matrix for all stats.

    :param dataset: Dataset of all documents and their content
    :return: document-stat matrix
    """

    if type(dataset) is not DataFrame:
        raise TypeError('Dataset must be a (Pandas) Dataframe')

    hyper_len = lambda links: len(links) if notnull(links) else 0

    # Make a copy, removing unwanted columns
    stat_matrix = dataset.drop(columns=['is_abusive', 'document_content', 'hyperlinks'])

    # Also calculate the length of the pre-processed document
    stat_matrix['processed_length'] = dataset['document_content'].apply(len)
    stat_matrix['num_links'] = dataset['hyperlinks'].apply(hyper_len)

    return SparseDataFrame(csr_matrix(stat_matrix), columns=stat_matrix.columns)
