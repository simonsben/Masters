from utilities.data_management import make_path, check_readable, rename_file, prepare_csv_writer
from re import compile, match
from unidecode import unidecode

# Generate and check path
base = make_path('data/datasets/100k-abusive-tweets/')
filename = '100k-abusive-tweets'

source_path = base / (filename + '_backup.csv')
dest_path = base / (filename + '.csv')

file_header = ['index', 'label', 'content']

if not source_path.exists():
    check_readable(dest_path)
    rename_file(dest_path, source_path)

source_file = source_path.open(mode='r', encoding='utf-8')
csv_writer, dest_file = prepare_csv_writer(dest_path, file_header)

regex = compile(r'(.+)\s(\w+)\s(\d+)[\n\r]')

index = 0
for doc in source_file:
    if doc == '\n':
        continue

    line = match(regex, doc)
    content = unidecode(line.group(1))
    document = [index, line.group(2), content]

    csv_writer.writerow(document)

    index += 1

dest_file.close()
