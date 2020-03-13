if __name__ == '__main__':
    from utilities.data_management import move_to_root, make_path, open_w_pandas, check_existence, \
        load_execution_params, make_dir, write_context_map, save_dataframe
    from model.extraction import split_into_contexts
    from model.expansion.intent_seed import tag_intent_documents
    from utilities.pre_processing import final_clean
    from pandas import DataFrame
    from numpy import savetxt, zeros, hstack, arange
    from numpy.random import choice

    move_to_root(5)

    params = load_execution_params()
    data_name = params['dataset']
    base_path = make_path('data/prepared_data/')
    data_path = base_path / (data_name + '_partial.csv')
    objective_path = base_path / 'wikipedia_corpus_reduced_partial.csv'
    dest_dir = make_path('data/processed_data/') / data_name / 'analysis' / 'intent'

    check_existence(data_path)
    make_dir(dest_dir)
    print('Config complete, starting initial mask computation.')

    raw_documents = open_w_pandas(data_path)
    raw_objective = open_w_pandas(objective_path)

    documents = raw_documents['document_content'].values
    objective = raw_objective['document_content'].values
    original_indexes = raw_documents.index.values
    objective_indexes = arange(-1, -(objective.shape[0] + 1), -1, dtype=int)

    documents = hstack([documents, objective])
    original_indexes = hstack([original_indexes, objective_indexes])
    print('Data loaded.')

    # Split documents into contexts
    document_contexts, context_map = split_into_contexts(documents, original_indexes)
    print('Contexts extracted, expanded', len(documents), 'to', len(document_contexts))

    intent_values, intent_frames = tag_intent_documents(document_contexts, params['n_threads'])
    print('Initial intent mask computed.')

    document_contexts = list(map(final_clean, document_contexts))
    document_contexts = DataFrame(document_contexts, columns=['contexts'])
    document_contexts['document_index'] = context_map[:, 0]
    document_contexts['context_index'] = context_map[:, 1]
    print('Assembled processed data.')

    intent_values[document_contexts['document_index'].values < 0] = 0

    shuffle_pattern = choice(document_contexts.shape[0], document_contexts.shape[0], replace=False)
    document_contexts = document_contexts.iloc[shuffle_pattern]
    intent_values = intent_values[shuffle_pattern]
    intent_frames = intent_frames[shuffle_pattern]
    print('Mixed data.')

    # Save initial mask, context mapping, and intent frame (mask values)
    intent_frames = DataFrame(intent_frames)
    intent_frames.to_csv(dest_dir / 'intent_frame.csv', header=False, index=False)
    savetxt(dest_dir / 'intent_mask.csv', intent_values, delimiter=',', fmt='%.1f')
    print('Saved masks, saving contexts.')

    save_dataframe(document_contexts, dest_dir / 'contexts.csv')                # Save contexts
    print('Contexts saved.')
