from utilities.data_management import make_path, check_readable, rename_file, prepare_csv_writer, prepare_csv_reader, \
    expand_csv_row_size
from re import compile
from unidecode import unidecode

expand_csv_row_size()
newline_regex = compile(r'(<br />)[\n\r]?n|[\n\r]n')

# Generate and check path
directory_filename = '../datasets/storm-front/'
filename = 'storm-front'
directory = make_path(directory_filename)
source_path = directory / (filename + '_backup.csv')
dest_path = directory / (filename + '.csv')

file_header = ['date', 'user', 'document_content']

# Backup file
if not source_path.exists():
    check_readable(dest_path)
    rename_file(dest_path, source_path)

# Initialize csv reader and writer
csv_writer, dest_file = prepare_csv_writer(dest_path, file_header)
csv_reader, source_file, _ = prepare_csv_reader(source_path, encoding='utf-8', has_header=False)

# Prepare rows
for ind, line in enumerate(csv_reader):
    line = line[3:]
    line[2] = newline_regex.sub(' ', unidecode(line[2]))

    csv_writer.writerow(line)

source_file.close()
dest_file.close()
