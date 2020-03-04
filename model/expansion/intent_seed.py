from spacy import load
from numpy import asarray, squeeze, logical_not, add, percentile, sum
from itertools import compress
from multiprocessing import Pool
from model.extraction import generate_context_matrix

# Token and dependency sets for detecting basic intent
desire_verb_tags = {'VB', 'VBG', 'VBP', 'VBZ'}
non_active_desire_tags = {'VBD', 'VBN'}

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


def identify_basic_intent(context):
    """ Determines if parsed document contains a sequence of term that indicate intent """
    context = parser(context)
    
    source, target, timing, desire_verb, action_verb = None, None, None, None, None
    intent_score = .5

    for token in context:
        if intent_score == .5:
            source, target, timing, desire_verb, action_verb = None, None, None, None, None

        if token.tag_ != 'TO': continue                                         # Check for TO
        if token.head is None or token.head.pos_ != 'VERB': continue            # Check for action verb
        if token.head.head is None or token.head.head.pos_ != 'VERB': continue  # Check for desire verb

        # Check if the verb tense is correct
        desire_verb = token.head.head
        action_verb = token.head
        if desire_verb.tag_ not in desire_verb_tags: continue
        elif desire_verb.tag_ in non_active_desire_tags:
            intent_score = 0 if intent_score == .5 else intent_score

        # Check if statement is personal
        pronouns = list(filter(
            lambda _token: _token.pos_ == 'PRON' and _token.text in first_person_pronouns,
            desire_verb.children
        ))
        if len(pronouns) < 1: continue
        source = pronouns[0].text

        # Get action target
        target = assemble_related_information(action_verb, target_dependencies, target_relations)
        # Get action timing
        timing = assemble_related_information(action_verb, timing_dependencies, timing_relations)

        # Check for negations
        negations = len(list(filter(lambda _token: _token.dep_ == 'neg', desire_verb.children)))
        questions = len(list(filter(
            lambda _token: _token.tag_ in question_tags or _token.text in question_indicators,
            desire_verb.children
        )))

        if negations % 2 != 0:
            intent_score = 0 if intent_score == .5 else intent_score
            continue
        elif questions > 0:
            intent_score = 0 if intent_score == .5 else intent_score
            continue

        # Contains positive intent
        intent_score = 1
        break

    desire_verb = desire_verb.text if desire_verb is not None else None
    action_verb = action_verb.text if action_verb is not None else None

    return intent_score, source, desire_verb, action_verb, target, timing


def worker_init(*props):
    """ Initialization function for Pool workers """
    global parser
    parser = load('en_core_web_md')


def tag_intent_documents(contexts, n_threads=None):
    """ Determines whether each context contains intent, then return intent value and base verb """
    if n_threads is None:
        from utilities.data_management import load_execution_params
        n_threads = load_execution_params()['n_threads']

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
