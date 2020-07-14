from utilities.data_management import make_path, make_dir, open_w_pandas, load_vector, check_existence
from spacy import load
from spacy.displacy import serve
from matplotlib.pyplot import show
from re import compile
from config import dataset

base = make_path('data/processed_data') / dataset / 'analysis' / 'intent'
context_path = base / 'contexts.csv'
cone_path = base / 'cone_mask.csv'
figure_path = make_path('figures') / dataset / 'analysis'

check_existence([cone_path, context_path])
make_dir(figure_path)

contexts = open_w_pandas(context_path)
cone_mask = load_vector(cone_path)

intent = cone_mask == 1
intent_contexts = contexts['contexts'].values[intent]

model = load('en_core_web_sm')
leading_regex = compile(r'[^ ]')

to_display = None
for context in intent_contexts:
    inp = input(context + '\n')
    if inp == 'y':
        first_non_space = leading_regex.search(context)
        to_display = context[first_non_space.start():]
        break

print('Visualizing', to_display)
serve(
    model(to_display),
    options={'compact': True}
)

show()
