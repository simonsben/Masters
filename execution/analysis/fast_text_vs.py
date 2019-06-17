from dask.dataframe import read_csv
from matplotlib.pyplot import subplots, title, show, tight_layout, savefig
from utilities.data_management import move_to_root, make_path, check_existence, check_writable, load_execution_params
from pandas import read_csv as p_read
from csv import QUOTE_NONE

move_to_root()

dataset = load_execution_params()['dataset']
base = make_path('data/lexicons/fast_text/')
raw_path = base / 'fast_text.vec'
dest_path = base / 'fast_text_desc.csv'
fig_path = make_path('figures/') / dataset / 'analysis' / 'fast_text_desc.png'

check_writable(fig_path)

if not dest_path.exists():
    print('Computing vector description')
    check_existence(raw_path)
    check_writable(dest_path)

    data = read_csv(raw_path, quoting=QUOTE_NONE, delimiter=' ', header=None, skiprows=1)
    description = data.describe().compute()
    description.to_csv(dest_path)
    print('Calculated description')
else:
    description = p_read(dest_path, quoting=QUOTE_NONE, index_col=0)
    print('Description loaded')

x = list(range(1, description.shape[1] + 1))
means = description.iloc[1]
stds = description.iloc[2]

fig, ax = subplots()
ax.scatter(x, means, 5)
ax.errorbar(x, means, yerr=stds, ls='none', elinewidth=2)

title('Distribution of FastText vectors')
ax.set_xlabel('Dimension')
ax.set_ylabel('Vector value')
ax.set_xlim(0, description.shape[1] + 1)
fig.set_size_inches(20, 5, forward=True)
tight_layout()

savefig(fig_path)

show()
