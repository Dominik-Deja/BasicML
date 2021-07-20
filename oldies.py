# No data tree
class Node:
    def __init__(self, left=None, right=None, curr_depth=None):
        self.left = left
        self.right = right
        self.curr_depth = curr_depth

    def __str__(self):
        return f'This node is at level: {self.curr_depth}'


def attrgetter(object, name, value=None):
    for element in name.split('.'):
        object = getattr(object, element, value)
    return object


def attrsetter(object, name, value):
    pre, inter, suf = name.rpartition('.')
    return setattr(attrgetter(object, pre) if pre else object, suf, value)


class DecisionTree:

    def __init__(self, data=None, target=None, max_depth=3):
        self.data = data
        self.target = target
        self.max_depth = max_depth
        self.root = Node()
        self.fitted_depth = 0

    def fit(self):
        # Check if it makes sense to grow a tree and if so then grow it
        attrsetter(self.root, 'left', Node(curr_depth=1))
        attrsetter(self.root, 'right', Node(curr_depth=1))
        self.fitted_depth = 1
        stack = ['left', 'right']
        while stack:
            name = stack.pop()
            curr_depth = name.count('.') + 2
            if curr_depth <= self.max_depth:  # Here you can add more conditions
                if self.fitted_depth < curr_depth:
                    self.fitted_depth = curr_depth
                attrsetter(self.root, f'{name}.left', Node(curr_depth=curr_depth))
                attrsetter(self.root, f'{name}.right', Node(curr_depth=curr_depth))
                if curr_depth < self.max_depth:
                    stack.append(f'{name}.left')
                    stack.append(f'{name}.right')

    def pretty_print(self):
        if self.fitted_depth == 0:
            return 'This tree is still a sapling. There\'s nothing to show'
        else:
            s = f'Tree with max fitted depth of {self.fitted_depth}:\n'
            s += 'root\n'
            stack = ['left', 'right']
            while stack:
                name = stack.pop()
                curr_depth = name.count('.') + 2
                s += f'{" "*curr_depth}{name}\n'
                if attrgetter(self.root,f'{name}.left'):
                    stack.append(f'{name}.left')
                if attrgetter(self.root,f'{name}.right'):
                    stack.append(f'{name}.right')
            return s

    def __str__(self):
        return self.pretty_print()


# Very old
class RecursiveTree:
    def __init__(self, max_depth=3, data=None, target_index=None):
        self.max_depth = max_depth
        self.root = None
        self.data = data
        self.target_index = target_index

    def attrgetter(self, object, name):
        for element in name.split('.'):
            object = getattr(object, element)
        return object

    def attrsetter(self, object, name, value):
        pre, inter, suf = name.rpartition('.')
        return setattr(self.attrgetter(object, pre) if pre else object, suf, value)
    def fit(self):
        self.root = self.grow_tree()

    def grow_tree(self, depth=0):
        if depth < self.max_depth:
            return Node(left=self.grow_tree(depth=depth+1),
                        right=self.grow_tree(depth=depth+1),
                        curr_depth=depth+1)
