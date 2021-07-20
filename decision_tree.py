# This is a basic decision tree
from sklearn import datasets

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


if __name__ == '__main__':
    iris = datasets.load_iris()
    tree = DecisionTree(data=iris.data, target=iris.target, max_depth=3)


    # tree = DecisionTree(max_depth=1)
    # print(tree.fitted_depth)
    # tree.fit()
    # print(tree.fitted_depth)
    # tree = DecisionTree(max_depth=3)
    # print(tree)
    # tree.fit()
    # print(tree)
    # print(tree.root.left)
    # print(tree.root.left.left)
    # print(tree.root.left.left.left)
    # print(tree.root.left.left.left.left)

    # tree = RecursiveTree()
    # tree.fit()
    # print(tree.root.left.left)
    # #print(attrgetter(tree, 'root.left.right'))
    # #attrsetter(tree, 'root.left.left.leftish', 999)
    # tree._attrsetter(tree, 'root.left.left.leftish', 999)
    # print(tree.root.left.left.leftish)



    # def fit(self):  # Would be great, but not working due to the lack of pointers in Python
    #     stack = [self.root]
    #     depth, max_depth = 0, 3
    #     while stack:
    #         node = stack.pop()  # I would like to pass it by reference
    #         node = Node()
    #         if split > 0:  # split_yes (dependent on depth and max_depth)
    #             node.left = None  # So that I'm adding attribute left to self.root (and further expanding)
    #             node.right = None
    #             stack.append(node.left)
    #             stack.append(node.right)
    #             print(stack)
    #             split += -1
