from utilities.analysis import svd_embeddings, get_nearest_neighbours
from scipy.cluster.vq import whiten, kmeans
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
from numpy import percentile, argmin
from nltk.corpus import wordnet

x_key, y_key = 'euclidean_distances', 'cosine_distances'


def cluster_neighbours(neighbours, refined=False, target=None):
    """ Calculate the cluster of embeddings around a given word """
    # Normalize data
    neighbours = neighbours.iloc[1:]
    normed_data = whiten(neighbours[[x_key, y_key]].values)

    # Threshold data
    e_distances = normed_data[:, 0]
    threshold = percentile(e_distances, 20)
    normed_data = normed_data[e_distances < threshold]

    # Cluster
    distortion = 2
    num_centroids = 1
    while distortion > (.25 if refined else 1):
        num_centroids += 1
        centroids, distortion = kmeans(normed_data, num_centroids)

    # Split data by centroid
    target_index = argmin([norm(centroid) for centroid in centroids])
    term_indexes = [
        ind + 1 for ind, point in enumerate(normed_data)
        if argmin([euclidean(centroid, point) for centroid in centroids]) == target_index
    ]

    close_bois = neighbours.iloc[term_indexes]
    if target is not None:
        close_bois['dist'] = close_bois.iloc[:, 1:-3].apply(lambda doc: euclidean(target, doc), axis=1)

    return list(neighbours['words'].iloc[term_indexes].values), close_bois[['words', 'euclidean_distances', 'cosine_distances', 'dist']]


def wordnet_expansion(lexicon, n_words=None):
    """ Expand lexicon using wordnet """
    if n_words is None:
        n_words = 5

    looker = wordnet.synsets
    expanded_lexicon = [
        [str(synonym.lemmas()[0].name()) for synonym in looker(word)[:n_words]]
        for word in lexicon
    ]

    return expanded_lexicon


def embedding_expansion(lexicon, embeddings, simple_expand=None):
    """
    Expand lexicon using trained word embeddings
    :param lexicon: List of word embeddings
    :param embeddings: Pandas DataFrame of words and their embeddings
    :param simple_expand: Specify number of terms to take 'blindly' from around the target word, (default, don't use)
    :return: List of words in the expanded lexicon
    """
    # Normalize embeddings and expand lexicon
    embeddings = svd_embeddings(embeddings)
    if not simple_expand:
        expanded_lexicon = [
            cluster_neighbours(
                get_nearest_neighbours(embeddings, word)[0]
            )
            for word in lexicon
        ]
    else:
        expanded_lexicon = []
        for ind, word in enumerate(lexicon):
            new_terms = get_nearest_neighbours(embeddings, word, n_words=(simple_expand + 1))[0]

            if len(new_terms) < 1:
                continue

            new_terms = new_terms['words'].values[1:]
            expanded_lexicon.append(new_terms)

            print('Adding', new_terms, ' - ', round((ind + 1) / len(lexicon) * 10000) / 100, '% complete')

    return expanded_lexicon


def expand_lexicon(lexicon, embeddings=None, simple_expand=None):
    """ Expand lexicon using either wordnet expansion or word embeddings """
    if embeddings is None:
        return wordnet_expansion(lexicon, simple_expand)
    else:
        return embedding_expansion(lexicon, embeddings, simple_expand)


def build_verb_tree(verb_model, labels=None):
    """
    Construct a tree from tokens clustered using agglomerative clustering
    :param verb_model: Agglomerative clustering model
    :param labels: Labels
    :return: Tree of lists (i.e. nexted lists forming a tree structure)
    """
    tree_joints = verb_model.children_
    num_children = verb_model.n_leaves_

    tree = {}
    for index, relation in enumerate(tree_joints):
        new_joint = []  # New sub-tree
        targets = []    # List of sub-trees (in list) to join new joint to

        for joint in relation:  # For each joint
            # Determine whether the joint is a leaf (label) index or a sub-tree index to join to
            if joint < num_children:
                new_joint.append(joint if labels is None else labels[joint])
            else:
                targets.append(joint - num_children)

        if len(targets) == 0:   # If sub-tree only has leaf nodes
            tree[index] = new_joint
        else:                   # If sub-tree connects to other sub-tree(s)
            for target in targets:
                new_joint.append(tree[target])
                tree.pop(target)
                tree[index] = new_joint

    return list(tree.values())[0]


def get_branch_leaves(verb_tree, target_labels):
    """
    Get labels from all leaves within the smallest sub-tree that contains all the target labels
    :param verb_tree: Tree of lists (i.e. nexted lists forming a tree structure)
    :param target_labels: Set of target labels
    :return: All leaf labels from sub-tree
    """
    if not isinstance(target_labels, set):
        target_labels = set(target_labels)

    target_terms = pull_leaves(verb_tree, target_labels)
    return target_terms


def extract_leaves(tree, collection, leaf_type=str):
    """
    Extract all leaf labels from tree of lists
    :param tree: Tree of lists (i.e. nexted lists forming a tree structure)
    :param collection: Set to collect leaf labels in
    :param leaf_type: Type of leaf labels (ex. string)
    :return: None (collects using pass-by-reference)
    """
    if isinstance(tree, leaf_type):
        collection.add(tree)
        return

    for sub_tree in tree:
        extract_leaves(sub_tree, collection, leaf_type)


def pull_leaves(tree, target_labels, leaf_type=str, extract_terms=True):
    """
    Find smallest sub-tree that contains all target labels
    :param tree: Tree of lists (i.e. nexted lists forming a tree structure)
    :param target_labels: Set of labels that sub-tree must contain
    :param leaf_type: Type of leaf labels
    :param extract_terms: Whether to extract the labels from the found sub-tree
    :return: sub-tree of lists or set of sub-tree labels
    """
    if isinstance(tree, leaf_type):   # If leaf
        return tree in target_labels
    elif isinstance(tree, set):       # If found sub-tree already
        return tree

    # Compute how many of the target labels are contained in the current sub-tree
    sub_tree_sum = 0
    for sub_tree in tree:
        tmp = pull_leaves(sub_tree, target_labels, leaf_type)
        if isinstance(tmp, set):
            return tmp
        sub_tree_sum += tmp

    # If current sub-tree doesn't contain all target labels
    if sub_tree_sum < len(target_labels):
        return sub_tree_sum

    if not extract_terms:
        return tree

    sub_tree_labels = set()
    extract_leaves(tree, sub_tree_labels, leaf_type)
    return sub_tree_labels


def check_for_labels(labels, target_labels, clean=True):
    labels = set(labels)
    target_labels = set(target_labels)
    bad_targets = set()

    for label in target_labels:
        if label not in labels:
            if clean:
                print('Removing', label, 'from target set')
                bad_targets.add(label)
            else:
                raise ValueError('Invalid label', label)

    return target_labels - bad_targets
