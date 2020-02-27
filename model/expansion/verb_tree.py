from utilities.data_management import split_embeddings
from model.analysis import cluster_verbs
from numpy import asarray


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
    """ Checks whether the target labels are in the current set of labels before searching tree """
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


def build_tree_and_collect_leaves(embeddings, target_labels, max_labels=75, mask=None):
    """
    Takes embeddings, performs clustering, builds a tree, then collects the leaves from the smallest sub-tree that
    contains the target labels
    :param embeddings: Matrix with the first column containing the labels and the remainder containing the vectors
    :param target_labels: Target labels for defining the sub-tree
    :param max_labels: Max number of labels to include in the tree
    :param mask: Mask for
    :return: model, leaves, labels
    """
    if mask is not None:
        embeddings = embeddings[mask]

    labels, vectors = split_embeddings(embeddings)

    model = cluster_verbs(vectors, max_verbs=max_labels)

    labels = labels[:model.n_leaves_]
    vectors = vectors[:model.n_leaves_]

    action_tree = build_verb_tree(model, labels)
    target_action_verbs = check_for_labels(labels, target_labels)
    leaves = get_branch_leaves(action_tree, target_action_verbs)

    return model, leaves, labels, vectors


def get_sub_tree(leaves, labels, vectors, max_verbs=75):
    sub_tree_mask = asarray([label in leaves for label in labels])
    sub_tree_vectors = vectors[sub_tree_mask]
    sub_tree_model = cluster_verbs(sub_tree_vectors, max_verbs=max_verbs)

    return sub_tree_model, sub_tree_mask
