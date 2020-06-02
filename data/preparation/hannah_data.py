from utilities.data_management import make_path
from os import rename
from pandas import read_csv
from unidecode import unidecode

base = make_path('data/datasets/hannah_data/')
ready_path = base / 'hannah_data.csv'
backup_path = base / 'hannah_data_backup.csv'

if not backup_path.exists():
    ready_path.rename(backup_path)


data = read_csv(backup_path)

data['document_content'] = data['document_content'].apply(unidecode)

data.to_csv(ready_path)
