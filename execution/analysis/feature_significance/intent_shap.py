from shap import GradientExplainer
from model.networks import generate_intent_network, load_model_weights
from model.layers.realtime_embedding import RealtimeEmbedding
from utilities.data_management import get_model_path, make_path, check_existence, make_dir, get_embedding_path, \
    open_w_pandas
from utilities.plotting import feature_significance, show, set_font_size
from config import dataset, max_tokens, embedding_dimension
from fasttext import load_model
from numpy import asarray, sum
from numpy.random import choice

target_contexts = asarray([
    'first of all i want to address the fact that you are an idiot',
    'if you tell us to pray quieter we ll kill you',
    'i ll wait you lying retard',
    'i know for a fact that if i had a kid and he was wearing those kind of clothing my reaction would be this one i ll kick your butt you little bastard',
    'we need segregation from these stupid filthy diseased savages',
    'thats nice i will paraphrase what st said blah blah blah i m a dirty f king jew blah blah',
    'don t refer to us as a bunch of hillbillies or we ll kick your ass',
    'one night when i disagreed with him he d grab me by the throat and said if you don t do what i say i will kill you',
    'i felt like saying if you touch me i will kill you',
    'obama isn t a leftist you ing nazi pig incestuous ing clown i ll rip your ing intestines out and feed them to dogs',
    'now go away or we ll kick your ass',
    'i ll ignore the troll are you bnp or anal',
    'if you come to me and threaten my life i will kill you',
    'we need to stop being soft hiding behind a wall of tolerance and start kicking some black and muslim ass',
    'we ll rape your wife pretoria give me your gun or i ll rape your wife',
    'we are fully aware of how high the humidity is so shut the hell up spend your money and get the hell out of here or we ll kick your ass',
    'i would recommend to say hey don t act a negro who are they for you',
    'but if you don t i will look for you i will find you and i will kill you',
    'those white idiots are begging her not to kill black babies i want to buy her a beer honestly if that represents christianity then i want no part of it',
    'those words just make me want to break things on you people like you make me sick'
])
num = len(target_contexts)

base = make_path('data/processed_data') / dataset / 'analysis' / 'intent'
context_path = base / 'contexts.csv'
weight_path = get_model_path('intent')
embedding_path = get_embedding_path()

figure_base = make_path('figures') / dataset / 'analysis' / 'shap'
figure_path_gen = lambda example_num: figure_base / ('intent_example_%d.png' % example_num)

check_existence([context_path, weight_path, embedding_path])
make_dir(figure_path_gen(0), max_levels=3)

contexts = open_w_pandas(context_path)['contexts'].values

# Generate model and load weights
model = generate_intent_network(max_tokens, embedding_dimension)
load_model_weights(model, weight_path)

fast_text_model = load_model(embedding_path)
realtime = RealtimeEmbedding(fast_text_model, contexts)

# Get random selection of contexts
selection = choice(len(contexts), 10000, replace=False)
embedded_data = realtime.embed_data(contexts[selection])

# Load the prepare the shap explainer
explainer = GradientExplainer(model, embedded_data)

set_font_size(16)
for index, context in enumerate(target_contexts[:num]):
    check_data = realtime.embed_data(asarray([context]))
    [shap_values] = explainer.shap_values(check_data)
    print(context)

    tokens = context.split(' ')
    token_values = sum(shap_values, axis=2).reshape(-1)

    package = list(reversed(list(zip(tokens, token_values))))
    feature_significance(package, 'SHAP values for intent model', figure_path_gen(index))

show()
