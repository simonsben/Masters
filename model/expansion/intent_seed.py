from spacy import load
from re import compile
from numpy import zeros, asarray, squeeze, logical_not, add, percentile
from sklearn.feature_extraction.text import CountVectorizer
from itertools import compress


intent_lead_terms = {'going', 'want', 'need', 'love', 'try',  'tempted', 'like', 'have', 'wish', 'got', 'hope',
                     'hoping', 'trying', 'gon', 'intend', 'wanted', 'tried', 'decided', 'ought', 'meaning'}
context_breaks = compile(r'[.?!]+')


# TODO clean up function
def identify_basic_intent(parsed):
    hits = []
    for ind, token in enumerate(parsed):
        if token.tag_ == 'TO' and token.head.pos_ == 'VERB' and token.head.head.pos_ == 'VERB':
            hits.append((token.head.head.text, token.text, token.head.text))
    return hits


def break_and_tag(documents, parser=None, lead_terms=intent_lead_terms):
    if parser is None:
        parser = load('en_core_web_sm')

    document_contexts = []
    intention_indexes = set()

    # For document in corpus
    for document in documents:
        # Get non-zero length contexts/sentences
        contexts = list(filter(
            lambda part: len(part) > 0,
            context_breaks.split(document)
        ))

        corpus_offset = len(document_contexts)
        document_contexts += contexts

        # For context in document
        for context_offset, context in enumerate(contexts):
            # Parse context
            parsed = parser(context)

            # Check for basic intent structure
            target = identify_basic_intent(parsed)
            if len(target) < 1:
                continue

            # For basic structure in context
            for hit in target:
                index = corpus_offset + context_offset

                # If leading variable is in the basic intent terms set
                if hit[0] in lead_terms:
                    intention_indexes.add(index)

    # Convert intention index list to boolean array
    intent = zeros(len(document_contexts), dtype=bool)
    intent[list(intention_indexes)] = True

    print('intent percentage', len(intention_indexes) / len(document_contexts))

    return document_contexts, intent


def get_intent_terms(documents):
    # Get contexts with intent
    document_contexts, has_intent = break_and_tag(documents)

    # Initialize document vectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=15000)

    # Construct document matrix
    document_matrix = vectorizer.fit_transform(document_contexts)
    features = vectorizer.get_feature_names()

    # Get mask for contexts without intent
    no_intent = logical_not(has_intent)

    # Get term sums for intent and unlabelled documents
    labeled_freq = squeeze(asarray(document_matrix[has_intent].sum(axis=0)))
    unlabelled_freq = squeeze(asarray(document_matrix[no_intent].sum(axis=0)))

    # Generate mask for the 10th percentile terms
    term_sums = add(labeled_freq, unlabelled_freq)
    threshold = percentile(term_sums, 10)
    min_mask = term_sums > threshold

    # Get number of labelled and unlabelled documents
    num_lab = labeled_freq.shape[0]
    num_unlab = unlabelled_freq.shape[0]

    # Assemble significant terms
    relevant_contexts = compress(zip(features, labeled_freq, unlabelled_freq), min_mask)
    significant_terms = []
    for term, lab_count, unlab_count in relevant_contexts:
        if unlab_count == 0:
            unlab_count = 1
        freq = (lab_count / num_lab) / (unlab_count / num_unlab)

        if freq > 1:
            significant_terms.append((term, freq))

    significant_terms = sorted(significant_terms, key=lambda term: term[1], reverse=True)

    return significant_terms
