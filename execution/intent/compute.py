if __name__ == '__main__':
    from utilities.data_management import make_path, open_w_pandas, check_existence, make_dir, save_dataframe, \
        vector_to_file
    from model.extraction import split_into_contexts
    from model.expansion.intent_seed import tag_intent_documents
    from utilities.pre_processing import final_clean
    from pandas import DataFrame
    from numpy import savetxt, zeros, hstack, arange, sum, logical_not, vstack
    from numpy.random import choice
    from time import time
    from config import dataset

    base_path = make_path('data/prepared_data/')
    data_path = base_path / (dataset + '_partial.csv')
    objective_path = base_path / 'wikipedia_corpus_reduced_partial.csv'
    dest_dir = make_path('data/processed_data/') / dataset / 'analysis' / 'intent'

    check_existence(data_path)
    make_dir(dest_dir)
    print('Config complete, starting initial mask computation.')

    empty_frame = [None, None, None, None, None, -1]

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
    document_contexts, (document_indexes, context_indexes) = split_into_contexts(documents, original_indexes)
    print('Contexts extracted, expanded', len(documents), 'to', len(document_contexts))

    document_contexts = DataFrame(list(map(final_clean, document_contexts)), columns=['contexts'])
    document_contexts['document_index'] = document_indexes
    document_contexts['context_index'] = context_indexes
    print('Assembled processed data.')

    non_wikipedia = document_contexts['document_index'].values >= 0
    unknown_contexts = document_contexts['contexts'].values[non_wikipedia]
    num_wikipedia = sum(logical_not(non_wikipedia))
    num_contexts = document_contexts.shape[0]

    start = time()
    intent_values, intent_frames = tag_intent_documents(unknown_contexts)
    print('Computed rough labels in', time() - start, 'seconds')

    # Add negative intent values and empty frames for wikipedia contexts
    intent_values = hstack([intent_values, zeros(num_wikipedia)])
    intent_frames = vstack([intent_frames, [empty_frame] * num_wikipedia])

    # Shuffle contexts, rough labels, and frames
    shuffle_pattern = choice(num_contexts, num_contexts, replace=False)
    document_contexts = document_contexts.iloc[shuffle_pattern]
    intent_values = intent_values[shuffle_pattern]
    intent_frames = intent_frames[shuffle_pattern]
    print('Data shuffled.')

    # Save initial mask, context mapping, and intent frame (mask values)
    intent_frames = DataFrame(intent_frames)
    intent_frames.to_csv(dest_dir / 'intent_frame.csv', header=False, index=False)
    vector_to_file(intent_values, dest_dir / 'intent_mask.csv', fmt='%.1f')
    print('Saved masks, saving contexts.')

    save_dataframe(document_contexts, dest_dir / 'contexts.csv')                # Save contexts
    print('Contexts saved.')
