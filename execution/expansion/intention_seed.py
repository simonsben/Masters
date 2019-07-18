from model.expansion.intent_seed import get_intent_terms
from utilities.data_management import move_to_root, make_path, open_w_pandas, check_existence, load_execution_params, \
    make_dir
from pandas import DataFrame

move_to_root()

params = load_execution_params()
data_name = '24k-abusive-tweets'
data_path = make_path('data/prepared_data/') / (data_name + '.csv')
dest_path = make_path('data/processed_data/') / data_name / 'analysis' / 'intent' / 'intent_seed.csv'

check_existence(data_path)
make_dir(dest_path)

content = open_w_pandas(data_path)['document_content']
print('Content loaded')

intent_terms = get_intent_terms(content)
intent_terms = DataFrame(intent_terms, columns=['terms', 'significance'])

print(intent_terms)

intent_terms.to_csv(dest_path)
