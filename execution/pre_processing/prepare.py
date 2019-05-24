from os import chdir, listdir
from re import compile, search


file_regex = compile(r'\.py$')
chdir('../../data/preparation')

files = listdir('.')

for file in files:
    if search(file_regex, file) is not None:
        print('Executing: ', file)

        with open(file) as fl:
            sub = exec(fl.read())
