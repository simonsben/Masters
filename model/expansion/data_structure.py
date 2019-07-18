from utilities.plotting.utilities import generate_3d_figure
from numpy import sum, max, array
from matplotlib.pyplot import show


def new_term(term, _parent=None):
    return term if type(term) is Term else Term(term, _parent)


def get_depth(level_map):
    if type(level_map) is not list or len(level_map) == 0:
        return 1
    return max([get_depth(entry) for entry in level_map]) + 1


class Term:
    def __init__(self, _term, _parent=None):
        self.term = _term
        self.parent = _parent

    def get_parent(self):
        return self.parent

    def __hash__(self):
        return hash(self.term)

    def __eq__(self, other):
        if isinstance(other, Term):
            return self.term == other.term
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.term)  # (str(self.parent) + '->' if self.parent is not None else '')


class Terms:
    def __init__(self, _init_set=None):
        self.terms = {new_term(term) for term in _init_set} if _init_set is not None else set()

    def add_term(self, _term, _parent=None):
        if _term not in self.terms:
            self.terms.add(new_term(_term, _parent))

    def add_terms(self, terms, _parent=None):
        for term in terms:
            self.add_term(term, _parent)

    def visualize(self, term_map, blocking=False):
        fig, ax = generate_3d_figure()

        points = []
        weights = []

        level_map = {}
        remaining = list(self.terms)

        last = 0
        while len(remaining) > 0:
            if len(remaining) == last:
                print('Never found', [str(rem) for rem in remaining])
                break
            last = len(remaining)

            to_remove = []
            for ind, r_term in enumerate(remaining):
                term = str(r_term)
                parent = str(r_term.parent) if r_term.parent is not None else None
                if parent is not None and parent not in level_map:
                    continue

                level_map[term] = level_map[parent] + 1 if parent in level_map else 0
                to_remove.append(ind)

                term_point = term_map[term]
                if parent is not None:
                    line_x, line_y, line_z = array([term_point, term_map[parent]]).transpose()
                    ax.plot(line_x, line_y, line_z)

                points.append(term_point)
                weights.append(level_map[term])

            to_remove.reverse()
            for ind in to_remove:
                remaining.pop(ind)

        points = array(points).transpose()
        weights = array(weights)
        print(points.shape, weights.shape)
        print(weights)

        x, y, z = points
        img = ax.scatter(x, y, z)
        # cbar = fig.colorbar(img, ax=ax)

        # ax.set_xlabel('Dimension 1')
        # ax.set_ylabel('Dimension 2')
        # ax.set_zlabel('Dimension 3')
        # cbar.ax.set_ylabel('Term depth')

        if blocking:
            show()


    def __str__(self):
        return str(self.terms)
