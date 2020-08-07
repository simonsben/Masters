from utilities.data_management import open_w_pandas, make_path, check_existence, get_dataset_name, make_dir
from utilities.plotting import hist_plot, show, set_font_size
from config import dataset

data_path = make_path('data/prepared_data') / (dataset + '_partial.csv')
figure_base = make_path('figures') / dataset / 'analysis'

check_existence(data_path)
make_dir(figure_base)

data = open_w_pandas(data_path)
get_length = lambda document: len(document) if isinstance(document, str) else 0

original_lengths = data['original_length'].values
processed_lengths = data['document_content'].apply(get_length).values

original_length = original_lengths.sum()
processed_length = processed_lengths.sum()

print('Raw length', original_length)
print('Processed length', processed_length)
print('Percentage removed', (original_length - processed_length) / original_length)

dataset_name = get_dataset_name()

set_font_size(16)
hist_plot(original_lengths, dataset_name + ' original document length', figure_base / 'original_length.png', x_angle=30)
hist_plot(processed_lengths, dataset_name + ' processed document length', figure_base / 'processed_length.png', x_angle=30)

show()
