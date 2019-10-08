from utilities.data_management import make_path, move_to_root

move_to_root()

base = make_path('data/prepared_data/')
source_name = 'storm-front-full'
dest_name = 'storm-front'

variants = ['', '_partial']
num_lines = 1000000
print('Prepared params, starting selection')

for variant in variants:
    source, dest = base / (source_name + variant + '.csv'), base / (dest_name + variant + '.csv.gz')

    dest_fl = dest.open('w')
    with source.open('r') as src_fl:
        for index, line in enumerate(src_fl):
            if index > num_lines:
                break
            dest_fl.write(line)
    dest_fl.close()

    print('Finished', variant)
print('All writes complete.')

