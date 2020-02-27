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



