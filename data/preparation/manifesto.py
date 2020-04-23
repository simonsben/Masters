from re import compile
from utilities.data_management import make_path
from os import rename

base_dir = make_path('data/datasets/manifesto/')
filename = base_dir / 'manifesto.csv'
backup_filename = base_dir / 'manifesto_backup.csv'

if not backup_filename.exists():
    rename(filename, backup_filename)

split_regex = compile(r'[\n\r]{2,}')

with open(backup_filename, 'r', encoding='utf-8') as fl:
    data = split_regex.split(fl.read())

data = [
    str(index) + ',"' + subsection.replace('\n', ' ').replace('"', '\'') + '"'
    for index, subsection in enumerate(data)
]
data = '\n'.join(data)

with open(filename, 'w', newline='', encoding='utf-8') as fl:
    fl.write(data)
