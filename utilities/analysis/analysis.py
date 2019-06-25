from numpy import where
from sklearn.metrics import log_loss


def get_feature_values(model, value_type='weight'):
    """ Pulls feature importance values from XGBoost model """
    return model.get_booster().get_score(importance_type=value_type)


def calculate_loss(predictions, labels, threshold):
    """ Calculates the binary-crossentropy loss of a set of predictions with a given threshold """
    thresh_preds = where(predictions > threshold, 1, 0)
    return log_loss(labels, thresh_preds)
