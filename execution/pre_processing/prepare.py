# Execute preparation scipts (see data/preparation)
from os import chdir, listdir
from re import compile, search

file_regex = compile(r'\.py$')      # Regex to identify Python files
chdir('../../data/preparation')     # Change working directory to the preparation folder
files = listdir('.')                # List all files in directory

for file in files:                              # For each preparation scipt
    if search(file_regex, file) is not None:    # If file is a python scipt (i.e. not the README)
        print('Executing: ', file)

        with open(file) as fl:                  # Open then execute file
            sub = exec(fl.read())
