from dask.dataframe import read_csv
from matplotlib.pyplot import subplots, title, show, tight_layout, savefig, close
from utilities.data_management import make_path, check_existence, check_writable
from pandas import read_csv as p_read, DataFrame
from csv import QUOTE_NONE
import config

dataset_name = config.dataset
lex_name = config.fast_text_model
base = make_path('data/lexicons/fast_text/')
raw_path = base / (lex_name + '.vec')
dest_path = base / (lex_name + '_desc.csv')
fig_path = make_path('figures/') / dataset_name / 'analysis' / 'fast_text_desc.png'

check_writable(fig_path)

if not dest_path.exists():
    print('Computing vector description')
    check_existence(raw_path)
    check_writable(dest_path)

    data = read_csv(raw_path, quoting=QUOTE_NONE, delimiter=' ', header=None, skiprows=1)
    description = DataFrame([
        data.mean().compute(),
        data.std().compute()
    ])
    description.to_csv(dest_path)
    print('Calculated description')
else:
    description = p_read(dest_path, quoting=QUOTE_NONE, index_col=0)
    print('Description loaded')

x = list(range(1, description.shape[1] + 1))
means = description.iloc[0]
stds = description.iloc[1]

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

# show()
close('all')
