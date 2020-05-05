from pandas import DataFrame
from numpy import zeros, min, max


def enforce_qualifying(qualifying, unlabelled, answers, label_key, window_size=5):
    """
    Determines user labels that were part of a window with a correctly labelled qualifying question

    :param DataFrame qualifying: DataFrame of user labels for qualifying contexts
    :param DataFrame unlabelled: DataFrame of user labels for unlabelled/unknown contexts
    :param list answers: List of *answers* to qualifying contexts
    :param str label_key: Name of the column with the labels
    :param int window_size: Number of unlabelled contexts passed to users for every qualifying question
    :return DataFrame: DataFrame of valid/qualified labels
    """
    # Ensure DataFrames are sorted
    qualifying = qualifying.sort_values(by='context_id')
    unlabelled = unlabelled.sort_values(by='context_id')

    # Collect labels associated with each user in the dataset
    user_labels = {}
    for index, user_id in enumerate(unlabelled['user_id'].values):
        if user_id not in user_labels:
            user_labels[user_id] = []

        user_labels[user_id].append(unlabelled.iloc[index].values)

    # Collect set of user labels with correctly answered qualifying questions
    qualified_answers = []
    for user_id in user_labels:
        user_answers = qualifying.loc[qualifying['user_id'].values == user_id]
        window_qualifier = user_answers[label_key].values == answers[:len(user_answers)]

        # For each qualifying window
        for window_index, label_window in enumerate(window_qualifier):
            if label_window:
                start = window_index * window_size
                end = start + window_size

                qualified_answers += user_labels[user_id][start:end]

    qualified_answers = DataFrame(qualified_answers, columns=unlabelled.columns)

    return qualified_answers


def count_labels(labels, index_key, label_key):
    """
    Aggregates the number of *votes* for a given label for each document

    :param DataFrame labels: array of labels collected from the dataset
    :param string index_key: Name of the column with the document indexes
    :param string label_key: Name of the column with the label values
    :return DataFrame: DataFrame of aggregated label counts for each document
    """
    # Determine size of array and options
    min_index, max_index = min(labels[index_key]), max(labels[index_key])
    options = sorted(set(labels[label_key]))
    option_map = {value: index for index, value in enumerate(options)}

    # Initialize array of counted values
    label_counts = zeros((max_index - min_index + 1, len(option_map)), dtype=int)

    # Count labels
    for label in labels[[index_key, label_key]].values:
        index, label_value = label
        label_counts[index, option_map[label_value]] += 1

    # Convert to DataFrame and compute equivalent 'labelled' value
    label_counts = DataFrame(label_counts, columns=options)
    label_counts['rating'] = label_counts[options[-1]] / label_counts[options[1:]].sum(axis=1)

    return label_counts
