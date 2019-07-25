if __name__ == '__main__':
    from utilities.data_management import move_to_root, make_path, open_w_pandas, check_existence, \
        load_execution_params, make_dir
    from execution.intent.prepare import pull_intent_terms
    from model.extraction import pull_document_contexts
    from model.expansion.term_learner import run_learning

    move_to_root()

    params = load_execution_params()
    data_name = '24k-abusive-tweets'
    data_path = make_path('data/prepared_data/') / (data_name + '.csv')
    dest_path = make_path('data/processed_data/') / data_name / 'analysis' / 'intent'

    check_existence(data_path)
    make_dir(dest_path)

    documents = open_w_pandas(data_path)['document_content'].iloc[:500]
    print('Content loaded')

    contexts, context_map = pull_document_contexts(documents)
    print('Contexts extracted')

    intent_terms, has_intent = pull_intent_terms(contexts, dest_path)
    print(intent_terms)

    expanded_terms = run_learning(contexts, ['gonna'])
