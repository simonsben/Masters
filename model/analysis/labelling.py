from pandas import DataFrame


def compute_user_index(data):
    """ Compute the label index specific to the user (i.e. first one they labelled, etc.) """


def enforce_qualifying(qualifying, unlabelled, answers, label_key, window_size=5):
    # Ensure DataFrames are sorted
    qualifying = qualifying.sort_values(by='context_id')
    unlabelled = unlabelled.sort_values(by='context_id')

    user_labels = {}
    for index, user_id in enumerate(unlabelled['user_id'].values):
        if user_id not in user_labels:
            user_labels[user_id] = []

        user_labels[user_id].append(unlabelled.iloc[index].values)

    qualified_answers = []
    for user_id in user_labels:
        user_answers = qualifying.loc[qualifying['user_id'].values == user_id]
        window_qualifier = user_answers[label_key].values == answers[:len(user_answers)]

        for window_index, label_window in enumerate(window_qualifier):
            if label_window:
                start = window_index * window_size
                end = start + window_size

                qualified_answers += user_labels[user_id][start:end]

    qualified_answers = DataFrame(qualified_answers, columns=unlabelled.columns)

    return qualified_answers
