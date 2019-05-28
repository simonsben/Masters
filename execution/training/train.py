from os import listdir
from re import compile, search


file_regex = compile(r'\.py$')
files = listdir('.')

exclusions = [
    'train.py',
    'train_deep_models.py'
]

for file in files:
    if search(file_regex, file) is not None and file not in exclusions:
        print('Executing:', file)

        with open(file) as fl:
            sub = exec(fl.read())

print('\nAll models trained.')
