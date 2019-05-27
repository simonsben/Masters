from utilities.data_management import check_existence, check_writable, make_path

base_directory = make_path('../data/lexicons/fast_text/')
source_filename = base_directory / 'fast_text.vec'
dest_filename = base_directory / 'fast_text_min.vec'

check_existence(source_filename)
check_writable(dest_filename)

with open(source_filename, 'r', encoding='utf-8') as s_fl:
    with open(dest_filename, 'w', encoding='utf-8') as d_fl:
        for i, line in enumerate(s_fl):
            if i > 1000:
                break

            d_fl.write(line)


