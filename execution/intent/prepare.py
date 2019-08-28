from model.expansion.intent_seed import get_intent_terms
from pandas import DataFrame
from numpy import savetxt


def pull_intent_terms(contexts, dir_path):
    intent_terms, has_intent = get_intent_terms(contexts)
    intent_terms = DataFrame(intent_terms, columns=['terms', 'significance'])

    contexts = DataFrame(contexts, columns=['contexts'])

    print('Terms extracted, saving mask, contexts, and terms')
    intent_terms.to_csv(dir_path / 'intent_terms.csv')
    contexts.to_csv(dir_path / 'contexts.csv')
    savetxt(dir_path / 'intent_mask.csv', has_intent, delimiter=',', fmt='%.1f')

    return intent_terms, has_intent
