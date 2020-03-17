from pandas import DataFrame
from re import compile, match
from numpy import min, max, ndarray, vectorize, mean, var

base_name_regex = compile(r'^[a-z]+')


def rescale_data(values):
    """ Linearly rescale data to (0, 1) """
    if not isinstance(values, ndarray):
        values = ndarray(values)
    if len(values.shape) > 1:
        raise TypeError('Expected values with shape (N,)')

    min_value = min(values)
    max_value = max(values)
    values += -min_value if min_value < 0 else min_value
    values /= (max_value - min_value)

    return values


def list_means(values):
    return apply_vectorizer(values, mean)


def list_variances(values):
    return apply_vectorizer(values, var)


def list_lengths(values):
    return apply_vectorizer(values, len)


def list_maxes(values):
    return apply_vectorizer(values, max)


def list_mins(values):
    return apply_vectorizer(values, min)


def apply_vectorizer(values, function):
    """ Vectorizes a function and applies it to a numpy array for faster execution """
    if not isinstance(values, list):
        raise TypeError('Expected list of values.')
    if not callable(function):
        raise TypeError('Expected function to be callable.')

    vectorizer = vectorize(function)
    return vectorizer(values)


def take_basics(dataset, display=False):
    """
    Takes set of basic statistics about a dataset.
    Assumes dataset to be a 2D table with numeric or string
    :param dataset: 2D Pandas dataset
    :param display: Whether the calculated summary should be printed to the console, (default False)
    :return: 2D basic parameters about the provided data
    """

    if type(dataset) is not DataFrame:
        raise TypeError('Dataset is the wrong type')

    data_summary = dataset.describe()

    # Add calculated columns
    comparator = dataset['original_length'] * 100

    data_summary['emoji_proportion'] = (dataset['emoji_count'] / comparator).describe()
    data_summary['special_proportion'] = (dataset['num_special'] / comparator).describe()
    data_summary['hashtag_proportion'] = (dataset['num_hashtags'] / comparator).describe()
    data_summary['upper_proportion'] = (dataset['upper_count'] / comparator).describe()

    correlations = {}
    for col in dataset:
        if 'proportion' in col:
            col_name = match(base_name_regex, col)
            correlations[col_name] = dataset[col].corr(dataset['is_abusive'])

    if display:
        for col in data_summary:
            print(str(data_summary[col]) + '\n')

    return data_summary, correlations
