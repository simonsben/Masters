from utilities.data_management import make_path
from os import chdir, listdir

# Stupid code I dont want to re-write that executes all the preparation scrips for the datasets I use

target_dir = make_path('data/preparation')

chdir(target_dir)                   # Change working directory to the preparation folder
files = listdir('.')                # List all files in directory

for file in files:                              # For each preparation script
    if '.py' not in file.__str__():    # If file is a python script (i.e. not the README)
        print('Executing: ', file)

        with open(file) as fl:                  # Open then execute file
            sub = exec(fl.read())
