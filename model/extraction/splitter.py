from pandas import DataFrame

non_features = ['is_abusive']


# CHECK double check this isnt used and remove
def splitter(dataset):
    """ Takes a pre-processed dataset and splits it into features and labels """

    if type(dataset) is not DataFrame:
        raise TypeError('dataset must be a (Pandas) DataFrame')

    headers = list(dataset.columns)

    labels = ['is_abusive']
    features = [feature for feature in headers if feature not in non_features]

    return dataset[features], dataset[labels]
