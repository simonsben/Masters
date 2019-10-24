from utilities.data_management import make_path, move_to_root
from pandas import read_csv

move_to_root()

base = make_path('data/prepared_data/')
source_name = 'storm-front-full'
dest_name = 'storm-front'

variants = ['', '_partial']
num_lines = 250000
print('Prepared params, starting selection')

for variant in variants:
    source_path, dest_path = base / (source_name + variant + '.csv'), base / (dest_name + variant + '.csv.gz')

    source = read_csv(source_path, index_col=0)
    sample = source.sample(n=num_lines)
    sample.to_csv(dest_path)

    # dest_fl = dest.open('w')
    # with source.open('r') as src_fl:
    #     for index, line in enumerate(src_fl):
    #         if index > num_lines:
    #             break
    #         dest_fl.write(line)
    # dest_fl.close()

    print('Finished', variant)
print('All writes complete.')
