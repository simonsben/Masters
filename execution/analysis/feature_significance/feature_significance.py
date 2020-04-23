from os import listdir
from utilities.data_management import make_path

base = make_path('execution/analysis/feature_significance')
files = listdir(base)

exclusions = ['feature_significance.py', 'deep_models.py', 'deep_shap_words.py']

for file in files:
    if file not in exclusions:
        print('Executing', file)

        with open(base / file) as fl:
            exec(fl.read())

