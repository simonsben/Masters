from utilities.data_management import make_path, check_readable, rename_file, prepare_csv_writer
from utilities.pre_processing import remove_unicode_values

# Generate and check path
directory_filename = '../datasets/storm-front/'
filename = 'storm-front'

directory = make_path(directory_filename)
source_path = directory / (filename + '_backup.csv')
dest_path = directory / (filename + '.csv')

file_header = ['label', 'content']

if not source_path.exists():
    check_readable(dest_path)
    rename_file(dest_path, source_path)

source_file = source_path.open(mode='r', encoding='utf-8')
csv_writer, dest_file = prepare_csv_writer(dest_path, file_header)

for document in source_file:
    csv_writer.writerow(remove_unicode_values(document))

dest_file.close()
