from config import n_threads, training_verbosity


def generate_tree_sequence_network():
    """ Generates sequence tree learner """
    from xgboost import XGBClassifier

    model = XGBClassifier(n_estimators=600, verbosity=training_verbosity, n_jobs=n_threads)

    return model
