from utilities.data_management import load_vector, open_w_pandas, make_path, make_dir, check_existence
from utilities.plotting import plot_training_statistics, show
from config import dataset

abuse_keys = ['val_accuracy', 'val_loss']
intent_keys = ['accuracy', 'loss']

base = make_path('data/processed_data') / dataset / 'analysis'
abuse_path = base / 'abuse' / 'training_history.csv'
intent_path = base / 'intent' / 'deep_history.csv'
figure_base = make_path('figures') / dataset / 'analysis'

check_existence([intent_path, abuse_path])
make_dir(figure_base)
print('Config complete.')

abuse = open_w_pandas(abuse_path)
intent = open_w_pandas(intent_path)
print('Loaded data.')

accuracy, loss = abuse[abuse_keys].values.transpose()
plot_training_statistics(accuracy, loss, 'Abuse training statistics', figure_base / 'abuse_training.png')

accuracy, loss = intent[intent_keys].values.transpose()
plot_training_statistics(accuracy, loss, 'Intent training statistics', figure_base / 'intent_training.png')
print('Finished plotting and saving figures.')

show()
