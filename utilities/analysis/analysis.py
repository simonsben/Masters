from numpy import where, min, max, std, mean, percentile, array
from sklearn.metrics import log_loss


def get_feature_values(model, value_type='weight'):
    """ Pulls feature importance values from XGBoost model """
    return model.get_booster().get_score(importance_type=value_type)


def calculate_loss(predictions, labels, threshold):
    """ Calculates the binary-crossentropy loss of a set of predictions with a given threshold """
    thresh_preds = where(predictions > threshold, 1, 0)
    return log_loss(labels, thresh_preds)


def length_stats(lengths):
    is_2d = type(lengths[0]) is list

    if not is_2d:
        return min(lengths), max(lengths), mean(lengths), std(lengths), percentile(lengths, 90), percentile(lengths, 95)
    return mean(array([length_stats(sub_lengths) for sub_lengths in lengths]), axis=0)
