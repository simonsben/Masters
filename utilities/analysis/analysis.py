def get_feature_values(model, value_type='weight'):
    return model.get_booster().get_score(importance_type=value_type)
