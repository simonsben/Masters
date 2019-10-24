from model.expansion.intent_seed import get_intent_terms, tag_intent_documents
from model.extraction.contexts import post_clean
from pandas import DataFrame
from numpy import savetxt


def pull_intent_terms(contexts, dir_path, full_docs=False):
    # Compute context mask and save
    intent_values = tag_intent_documents(contexts)
    filename = dir_path / (('document_' if full_docs else '') + 'intent_mask.csv')
    savetxt(filename, intent_values, delimiter=',', fmt='%.1f')
    print('Mask computed, computing intent terms')

    # Compute intent terms
    intent_terms, intent_values = get_intent_terms(contexts, intent_values=intent_values)
    print('Intent terms computed, saving')

    # Convert contexts to dataframe and save
    if not full_docs:
        contexts = DataFrame(contexts, columns=['contexts'])
        contexts['contexts'] = contexts['contexts'].map(lambda context: post_clean.sub('', context))
        contexts.to_csv(dir_path / 'contexts.csv')
        contexts = None

    # Convert intent terms to dataframe and save
    intent_terms = DataFrame(intent_terms, columns=['terms', 'significance'])
    intent_terms.to_csv(dir_path / 'intent_terms.csv')

    return intent_terms, intent_values
