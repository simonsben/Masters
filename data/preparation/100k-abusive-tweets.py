from utilities.data_management import make_path, check_readable, rename_file, prepare_csv_writer
from utilities.pre_processing import remove_unicode_values
from re import compile, match

# Generate and check path
directory_filename = '../datasets/100k-abusive-tweets/'
filename = '100k-abusive-tweets'

directory = make_path(directory_filename)
source_path = directory / (filename + '_backup.csv')
dest_path = directory / (filename + '.csv')

print(source_path, dest_path)

file_header = ['label', 'content']

if not source_path.exists():
    check_readable(dest_path)
    rename_file(dest_path, source_path)

source_file = source_path.open(mode='r', encoding='utf-8')
csv_writer, dest_file = prepare_csv_writer(dest_path, file_header)

regex = compile(r'(.+)\s(\w+)\s(\d+)[\n\r]')


for doc in source_file:
    if doc == '\n':
        continue
    line = match(regex, doc)
    document = remove_unicode_values([line.group(1), line.group(2)])

    csv_writer.writerow(document)

dest_file.close()
