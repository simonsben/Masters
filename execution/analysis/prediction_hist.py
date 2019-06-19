from utilities.data_management import make_path, open_w_pandas, move_to_root, load_execution_params, check_writable, \
    check_existence
from pandas import concat
from matplotlib.pyplot import show, tight_layout, savefig

move_to_root()

params = load_execution_params()
dataset_name = params['dataset']
files = ['test', 'train']
data_base = make_path('data/predictions/') / dataset_name
fig_path = make_path('figures/') / dataset_name / 'analysis/prediction_hist.png'

for file in files:
    check_existence(data_base / (file + '.csv'))
check_writable(fig_path)


data = concat([
    open_w_pandas(data_base / (file + '.csv')) for file in files
])
data.drop(columns='stacked', inplace=True)
print('Data loaded')

data.hist(bins=25, grid=False, sharex=True, figsize=(20, 15))
tight_layout()

savefig(fig_path)

show()
