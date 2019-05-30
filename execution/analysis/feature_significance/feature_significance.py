from os import listdir
from utilities.data_management import move_to_root, make_path

base = make_path('execution/analysis/feature_significance')
files = listdir('.')
move_to_root(4)

exclusions = ['feature_significance.py']

for file in files:
    if file not in exclusions:
        print('Executing', file)

        with open(base / file) as fl:
            exec(fl.read())

