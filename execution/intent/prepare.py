from model.expansion.intent_seed import get_intent_terms
from pandas import DataFrame


def pull_intent_terms(contexts, dest_path):
    intent_terms, has_intent = get_intent_terms(contexts)
    intent_terms = DataFrame(intent_terms, columns=['terms', 'significance'])

    print('Terms extracted, saving')
    intent_terms.to_csv(dest_path)

    return intent_terms, has_intent
