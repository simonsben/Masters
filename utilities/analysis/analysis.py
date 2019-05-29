from numpy import array


def get_feature_values(model, value_type='weight'):
    values = model.get_booster().get_score(importance_type=value_type)



    return array(sorted(
        [(int(key[1:]), values[key]) for key in values],
        key=lambda feat: feat[0]
    ))
