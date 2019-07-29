if __name__ == '__main__':
    from utilities.data_management import move_to_root, make_path, open_w_pandas, check_existence, \
        load_execution_params, make_dir
    from execution.intent.prepare import pull_intent_terms
    from model.extraction import pull_document_contexts
    from model.expansion.term_learner import run_learning
    from pandas import DataFrame

    move_to_root()

    params = load_execution_params()
    data_name = params['dataset']
    data_path = make_path('data/prepared_data/') / (data_name + '_partial.csv')
    dest_dir = make_path('data/processed_data/') / data_name / 'analysis' / 'intent'

    check_existence(data_path)
    make_dir(dest_dir)

    documents = open_w_pandas(data_path)['document_content']
    print('Content loaded')

    contexts, context_map = pull_document_contexts(documents)
    print('Contexts extracted')
    print('Initial', len(documents), 'split', len(contexts))

    intent_terms, has_intent = pull_intent_terms(contexts, dest_dir)
    print(intent_terms)

    expanded_terms = run_learning(contexts, intent_terms)
    intent_terms = DataFrame(intent_terms, columns=['terms', 'significance'])

    intent_terms.to_csv(dest_dir / 'expanded_intent_terms.csv')

