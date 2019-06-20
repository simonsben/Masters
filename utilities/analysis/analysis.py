def get_feature_values(model, value_type='weight'):
    """ Pulls feature importance values from XGBoost model """
    return model.get_booster().get_score(importance_type=value_type)
