from spacy import load
from numpy import asarray, squeeze, logical_not, add, percentile, sum
from itertools import compress
from multiprocessing import Pool
from model.extraction import generate_context_matrix
from collections.abc import Iterable
from config import n_threads

# Token and dependency sets for detecting basic intent
desire_verb_tags = {'VB', 'VBG', 'VBP', 'VBZ'}
non_active_desire_tags = {'VBD', 'VBN'}
special_auxiliaries = {'will', 'must', 'll'}
special_transforms = {'ll': 'will'}

first_person_pronouns = {'i', 'we', 'me', 'us', 'em', 'mine', 'myself', 'ourselves'}

target_dependencies = {'dobj', 'ccomp', 'acomp', 'xcomp'}
target_relations = {'det', 'compound', 'poss'}
timing_dependencies = {'npadvmod'}
timing_relations = {'nmod', 'compound'}

question_tags = {'WRB', 'WP'}
question_indicators = {'if', 'do'}


def assemble_related_information(base, base_dependency, information_dependency):
    potentials = list(filter(lambda _token: _token.dep_ in base_dependency, base.children))
    if len(potentials) > 0:
        information_indexes = [_token.i for _token in potentials[0].children if _token.dep_ in information_dependency]
        information = '"' + ','.join((str(index) for index in information_indexes + [potentials[0].i])) + '"'

        return information
    return None


def identify_basic_intent(context, index=-1):
    """ Determines if parsed document contains a sequence of term that indicate intent """
    if isinstance(context, str):
        context = parser(context)
    if not isinstance(context, Iterable):
        context = []

    source, target, timing, desire_verb, action_verb, short_desire = None, None, None, None, None, None
    intent_score = .5

    for token in context:
        if intent_score == .5:
            source, target, timing, desire_verb, action_verb, short_desire = None, None, None, None, None, None

        # Define base verb for check
        if token.pos_ != 'VERB': continue
        action_verb = token

        # Check auxiliaries of base verb to see if its a short or long intent case
        auxiliaries = [child for child in action_verb.children if child.dep_ == 'aux']
        if len(auxiliaries) <= 1: continue                                                  # No auxiliaries
        elif auxiliaries[0].pos_ == 'VERB' and auxiliaries[0].text in special_auxiliaries:  # Short case
            desire_verb = action_verb
            short_desire = auxiliaries[0]

            if short_desire.tag_ != 'MD': continue

        elif auxiliaries[0].pos_ != 'PART': continue                                        # Not long case
        elif auxiliaries[0].pos_ == 'PART' and action_verb.head is not None:                # Long case
            if action_verb.head.pos_ != 'VERB': continue
            desire_verb = action_verb.head

        # Check for pronouns
        pronouns = [
            child for child in desire_verb.children
            if child.pos_ == 'PRON' and child.i < desire_verb.i
        ]
        if len(pronouns) < 1: continue

        # Check for first person pronouns
        valid_pronouns = [pronoun for pronoun in pronouns if pronoun.text in first_person_pronouns]
        source = pronouns[0].text
        if len(valid_pronouns) < 1:
            intent_score = 0
            continue

        # Check tense of desire verb
        if desire_verb.tag_ not in desire_verb_tags:
            intent_score = 0
            continue

        # Get action target
        target = assemble_related_information(action_verb, target_dependencies, target_relations)
        # Get action timing
        timing = assemble_related_information(action_verb, timing_dependencies, timing_relations)

        # Check for negations
        # TODO ?only mark negation if between pronoun and desire?
        negations = len(list(filter(lambda _token: _token.dep_ == 'neg', desire_verb.children)))
        questions = len(list(filter(
            lambda _token: _token.tag_ in question_tags or _token.text in question_indicators,
            desire_verb.children
        )))

        # Check for non active desire, negations, or questions
        if desire_verb.tag_ in non_active_desire_tags:
            intent_score = 0
            continue
        elif negations % 2 != 0:
            intent_score = 0
            continue
        elif questions > 0:
            intent_score = 0
            continue

        # Contains positive intent
        intent_score = 1
        break

    # If short case, move desire verb for export
    if short_desire is not None:
        desire_verb = short_desire

    desire_verb = desire_verb.text if desire_verb is not None else None
    action_verb = action_verb.text if action_verb is not None else None

    if desire_verb is not None and desire_verb in special_transforms:
        desire_verb = special_transforms[desire_verb]

    return intent_score, source, desire_verb, action_verb, target, timing, index


def worker_init(*props):
    """ Initialization function for Pool workers """
    global parser
    parser = load('en_core_web_sm')


def tag_intent_documents(contexts):
    """ Determines whether each context contains intent, then return intent value and base verb """
    # Initialize worker pool
    worker_pool = Pool(n_threads, initializer=worker_init)

    # Process documents
    intent_data = worker_pool.map(identify_basic_intent, contexts)

    # Close pool
    worker_pool.close()
    worker_pool.join()

    # Split intent values and base verbs
    intent_data = asarray(intent_data)
    intent_values = intent_data[:, 0].astype(float)
    intent_frame = intent_data[:, 1:]

    print('intent percentage', sum(intent_values == 1) / len(intent_values))

    return intent_values, intent_frame


def get_intent_terms(contexts, intent_values=None, content_data=None):
    """ Computes intention terms """
    if intent_values is None:
        # Get contexts with intent
        intent_values = tag_intent_documents(contexts)
        print('Mask computed, running doc matrix', intent_values)

    intent_mask = intent_values == 1
    document_matrix, features = generate_context_matrix(contexts) if content_data is None else content_data

    # Get mask for contexts without intent
    no_intent_mask = logical_not(intent_mask)

    # Get term sums for intent and unlabelled documents
    labeled_freq = squeeze(asarray(document_matrix[intent_mask].sum(axis=0)))
    unlabelled_freq = squeeze(asarray(document_matrix[no_intent_mask].sum(axis=0)))

    # Generate mask for the 10th percentile terms
    term_sums = add(labeled_freq, unlabelled_freq)
    threshold = percentile(term_sums, 10)
    frequency_mask = term_sums > threshold

    # Get number of labelled and unlabelled documents
    num_labelled = labeled_freq.shape[0]
    num_unlabelled = unlabelled_freq.shape[0]

    # Assemble significant terms
    relevant_contexts = compress(zip(features, labeled_freq, unlabelled_freq), frequency_mask)
    significant_terms = []
    for term, lab_count, unlabelled_count in relevant_contexts:
        if unlabelled_count == 0:
            unlabelled_count = 1

        freq = (lab_count / num_labelled) / (unlabelled_count / num_unlabelled)
        significant_terms.append((term, freq))

    significant_terms = list(filter(lambda term: term[1] > 1, significant_terms))
    significant_terms = sorted(significant_terms, key=lambda term: term[1], reverse=True)

    return significant_terms, intent_values
