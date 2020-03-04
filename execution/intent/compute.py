if __name__ == '__main__':
    from utilities.data_management import move_to_root, make_path, open_w_pandas, check_existence, \
        load_execution_params, make_dir, write_context_map, save_dataframe
    from model.extraction import split_into_contexts
    from model.expansion.intent_seed import tag_intent_documents
    from utilities.pre_processing import final_clean
    from pandas import DataFrame
    from numpy import savetxt, zeros, vstack, arange
    from numpy.random import choice

    move_to_root()

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

    documents = vstack([documents, objective])
    original_indexes = vstack([original_indexes, objective_indexes])
    print('Data loaded.')

    # Split documents into contexts
    document_contexts, context_map = split_into_contexts(documents, original_indexes)
    print('Contexts extracted, expanded', len(documents), 'to', len(document_contexts))

    intent_values, intent_frames = tag_intent_documents(document_contexts, params['n_threads'])
    print('Initial intent mask computed.')

    array_index = 0
    for index in context_map:
        if index < 0:
            start = context_map[index].start
            stop = context_map[index].stop + 1
            intent_values[start:stop] = 0

    shuffle_pattern = choice(documents.shape[0], documents.shape[0], replace=False)
    documents = documents[shuffle_pattern]
    original_indexes = original_indexes[shuffle_pattern]

    # Save initial mask, context mapping, and intent frame (mask values)
    write_context_map(dest_dir / 'context_map.csv', context_map)
    savetxt(dest_dir / 'intent_mask.csv', intent_values, delimiter=',', fmt='%.1f')
    savetxt(dest_dir / 'intent_frame.csv', intent_frames, delimiter=',', fmt='%s')
    print('Saved masks, saving contexts.')

    document_contexts = list(map(final_clean, document_contexts))               # Clean contexts
    document_contexts = DataFrame(document_contexts, columns=['contexts'])      # Convert to DataFrame
    save_dataframe(document_contexts, dest_dir / 'contexts.csv')                # Save contexts
