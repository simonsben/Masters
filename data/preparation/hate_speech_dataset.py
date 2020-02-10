from os import listdir
from pandas import read_csv, DataFrame
from utilities.data_management import move_to_root, make_path, check_existence

move_to_root()

base_dir = make_path('data/datasets/hate_speech_dataset')
data_dir = base_dir / 'all_files'
metadata_path = base_dir / 'annotations_metadata.csv'

check_existence(metadata_path)

metadata = read_csv(metadata_path).values

label_map = {
    'noHate': 0,
    'hate': 1
}
file_ids = {
    meta[0]: label_map[meta[-1]]
    for meta in metadata if meta[-1] in label_map
}

documents = []
for filename in listdir(data_dir):
    path = data_dir / filename
    base_name = path.stem

    if base_name not in file_ids:
        continue

    with path.open(encoding='utf-8') as fl:
        documents.append(
            [file_ids[base_name], fl.read()]
        )

documents = DataFrame(documents, columns=['is_abusive', 'document_content'])
documents.to_csv(base_dir / 'hate_speech_dataset.csv')

print(documents)
