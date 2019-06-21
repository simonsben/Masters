from os import listdir
from utilities.data_management import move_to_root, make_path

move_to_root(4)

base = make_path('execution/analysis/feature_significance')
files = listdir(base)

exclusions = ['feature_significance.py', 'deep_models.py']

for file in files:
    if file not in exclusions:
        print('Executing', file)

        with open(base / file) as fl:
            exec(fl.read())

