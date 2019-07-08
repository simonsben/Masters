from utilities.data_management import move_to_root, make_path, check_existence, check_writable, open_exp_lexicon, \
    load_execution_params

move_to_root()

params = load_execution_params()
dataset_name = params['dataset']
embed_name = params['fast_text_model']
lexicon_name = 'anger'

expanded_path = make_path('data/processed_data') / dataset_name / 'analysis' / 'lexicon_expansion' / \
                (lexicon_name + '_' + embed_name + '.csv')
dest_path = make_path('data/prepared_lexicon/') / (lexicon_name + '_' + embed_name + '.csv')

check_existence(expanded_path)
check_writable(dest_path)

lexicon = open_exp_lexicon(expanded_path)
lexicon.to_csv(dest_path)

print('Lexicon saved')
print(lexicon)
