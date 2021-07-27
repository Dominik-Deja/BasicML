# This is a basic decision tree
from __future__ import annotations
from typing import Any, NoReturn, Type
from sklearn import datasets
import numpy as np

# Q: Shouldn't attrgetter and attrsetter be methods in DecisionTree?

def attrgetter(obj: object, name: str, value: Any = None) -> Any:
    """
    Returns value from nested objects/chained attributes (basically getattr() on steroids)
    :param obj: Primary object
    :param name: Path to the attribute (dot separated)
    :param value: Default value returned if a function fails to find the requested attribute value
    :return:
    """
    for attribute in name.split('.'):
        obj = getattr(obj, attribute, value)
    return obj


def attrsetter(obj: object, name: str, value: Any) -> NoReturn:
    """
    Sets the value of the attribute of a (nested) object (basically setattr() on steroids)
    :param obj: Primary object
    :param name: Path to the attribute (dot separated)
    :param value: Value to be set
    """
    pre, _, post = name.rpartition('.')
    setattr(attrgetter(obj, pre) if pre else obj, post, value)


class Node:
    """
    Building block of each tree
    May contain children nodes (left and right) or be a final node (called "leaf")
    """
    def __init__(self, data: np.ndarray = None, target: np.ndarray = None, left: Node = None, right: Node = None,
                 curr_depth: int = None, variable: int = None, threshold: float = None, leaf_value: np.ndarray = None) -> NoReturn:
        """
        :param data: NxM (where N denotes #observations and M denotes #variables) numpy array containing independent variables
        :param target: numpy vector containing dependent variable
        :param left: (if exists) node containing observations smaller than a given threshold at a given variable
        :param right: (if exists) node containing observations bigger than a given threshold at a given variable
        :param curr_depth: number of parent nodes directly above the current node
        :param variable: variable used to split data
        :param threshold: threshold at which data was split
        :param leaf_value: (if a node is a leaf, i.e. a final node with no children) the most frequent value(s) of a
                           dependent variable in a given node
        """
        self.data = data
        self.target = target
        self.left = left
        self.right = right
        self.curr_depth = curr_depth
        self.variable = variable
        self.threshold = threshold
        self.leaf_value = leaf_value

    def __str__(self):
        return f'This node is at level: {self.curr_depth}'


class DecisionTree:
    """
    Classification decision tree
    """
    def __init__(self, data: np.ndarray = None, target: np.ndarray = None, max_depth: int = 3):
        """
        :param data: NxM (where N denotes #observations and M denotes #variables) numpy array containing independent variables
        :param target: numpy vector containing dependent variable
        :param max_depth: maximum depth of a tree
        """
        self.data = data
        self.target = target
        self.max_depth = max_depth
        self.root = Node()
        self.fitted_depth = 0

    def __str__(self):
        if self.fitted_depth == 0:
            return 'This tree is still a sapling. There\'s nothing to show'
        else:
            s = f'Tree with max fitted depth of {self.fitted_depth}:\n'
            s += f'root ::: Split at variable {self.root.variable} at {self.root.threshold}\n'
            stack = ['left', 'right']
            while stack:
                name = stack.pop()
                curr_depth = name.count('.') + 2
                if attrgetter(self.root, f'{name}.variable'):
                    s += f"{' ' * curr_depth}{name} ::: Split at variable {attrgetter(self.root, f'{name}.variable')} at " \
                         f"{attrgetter(self.root, f'{name}.threshold')} (" \
                         f"{np.unique(attrgetter(self.root, f'{name}.target'), return_counts=True)[1]})\n"
                else:
                    s += f"{' ' * curr_depth}{name} ({np.unique(attrgetter(self.root, f'{name}.target'), return_counts=True)[1]}" \
                         f" and the leaf value is {attrgetter(self.root, f'{name}.leaf_value')})\n"
                if attrgetter(self.root,f'{name}.left'):
                    stack.append(f'{name}.left')
                if attrgetter(self.root,f'{name}.right'):
                    stack.append(f'{name}.right')
            return s

    @staticmethod
    def entropy(x: np.ndarray) -> float:
        if x.size == 0:
            return 0
        else:
            counts = np.unique(x, return_counts=True)[1]
            norm_counts = counts / counts.sum()
            return -(norm_counts * np.log(norm_counts)).sum()

    def information_gain(self, parent: np.ndarray, left_child: np.ndarray, right_child: np.ndarray) -> float:
        return self.entropy(parent) - (left_child.size / parent.size * self.entropy(left_child) +
                                       right_child.size / parent.size * self.entropy(right_child))

    @staticmethod
    def moving_average(x: np.ndarray, w: int) -> np.ndarray:
        return np.convolve(x, np.ones(w), 'valid') / w

    def find_best_split(self, data, target):
        best_split = {'variable' : None,
                      'threshold': None,
                      'gain': -1}
        if np.unique(target).size == 1:
            return best_split
        for variable in range(data.shape[1]):
            indices = data[:, variable].argsort()
            # Threshold is set to be a point in between two values (in a monotonically increasing set of unique values)
            thresholds = self.moving_average(data[indices, variable], 2)
            for threshold in thresholds:
                left_indices = data[:, variable] < threshold # TODO: Clean it, if possible, as it adds unnecessary complexity
                gain = self.information_gain(target, target[left_indices], target[np.invert(left_indices)])
                if gain > best_split['gain']:
                    best_split['variable'] = variable
                    best_split['threshold'] = threshold
                    best_split['gain'] = gain
        print(best_split, np.unique(target, return_counts=True)[1])
        return best_split
        ## This approach results in errors as it iterates through identical values, e.g. [1,1,2,2,2,3,4],
        ## gets different gain for identical values (as it increases/decreases step by step)
        ## and sets unreliable thresholds
        # for variable in range(data.shape[1]):
        #     indices = data[:, variable].argsort()
        #     target_sorted = target[indices]
        #     #data_sorted = data[indices,variable]
        #     for i, index in enumerate(indices):
        #         print(f'Information gain for variable {variable} at value {data[index, variable]} equals \
        #                {self.information_gain(target_sorted, target_sorted[:i], target_sorted[i:])}')
        #         gain = self.information_gain(target_sorted, target_sorted[:i], target_sorted[i:])
        #         if gain > best_split['gain']:
        #             best_split['variable'] = variable
        #             best_split['threshold'] = data[index,variable]
        #             best_split['gain'] = gain
        # print(best_split, np.unique(target, return_counts=True)[1])
        # return best_split
        #
        ## Pairplot
        # data_frame = pd.DataFrame(data=data)
        # data_frame['target'] = target
        # plot = sns.pairplot(data_frame, hue='target', corner=True)
        ## Just a sanity check plot for iris
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # fig.suptitle('Horizontally stacked subplots')
        # ax1.scatter(data[:, best_split['variable']], data[:, 0], c=target)
        # if best_split['variable'] == 0:
        #     plt.axvline(x=best_split['threshold'], linestyle='--')
        # ax2.scatter(data[:, best_split['variable']], data[:, 1], c=target)
        # if best_split['variable'] == 1:
        #     plt.axvline(x=best_split['threshold'], linestyle='--')
        # ax3.scatter(data[:, best_split['variable']], data[:, 2], c=target)
        # if best_split['variable'] == 2:
        #     plt.axvline(x=best_split['threshold'], linestyle='--')
        # ax4.scatter(data[:, best_split['variable']], data[:, 3], c=target)
        # if best_split['variable'] == 3:
        #     plt.axvline(x=best_split['threshold'], linestyle='--')
        # plt.show()

    def fit(self):
        # TODO: Check if passed slices are copied by value or by reference
        # TODO: Exception handling:
        #  (1) tree was already fitted
        #  (2) wrong data type
        #  (3) not enough training examples
        #  (4) data contains only one class
        best_split = self.find_best_split(self.data, self.target)
        left_indices = self.data[:,best_split['variable']] < best_split['threshold']
        self.root.variable = best_split['variable']
        self.root.threshold = best_split['threshold']
        attrsetter(self.root, 'left', Node(data=self.data[left_indices,:],
                                           target=self.target[left_indices],
                                           curr_depth=1))
        attrsetter(self.root, 'right', Node(data=self.data[np.invert(left_indices),:],
                                            target=self.target[np.invert(left_indices)],
                                            curr_depth=1))
        self.fitted_depth = 1
        stack = ['left', 'right']
        while stack:
            name = stack.pop()
            curr_depth = name.count('.') + 2
            data = attrgetter(self.root, f'{name}.data')
            target = attrgetter(self.root, f'{name}.target')

            best_split = self.find_best_split(data=data, target=target)
            if curr_depth <= self.max_depth and np.unique(target).size > 1 and best_split['gain'] > 0:
                if self.fitted_depth < curr_depth:
                    self.fitted_depth = curr_depth
                left_indices = data[:, best_split['variable']] <= best_split['threshold']
                attrsetter(self.root, f'{name}.variable', best_split['variable'])
                attrsetter(self.root, f'{name}.threshold', best_split['threshold'])
                attrsetter(self.root, f'{name}.left', Node(data=data[left_indices, :],
                                                           target=target[left_indices],
                                                           curr_depth=curr_depth))
                attrsetter(self.root, f'{name}.right', Node(data=data[np.invert(left_indices), :],
                                                            target=target[np.invert(left_indices)],
                                                            curr_depth=curr_depth))
                stack.append(f'{name}.left')
                stack.append(f'{name}.right')
            else:
                target_values, target_counts = np.unique(target, return_counts=True)
                attrsetter(self.root, f'{name}.leaf_value', target_values[target_counts == target_counts.max()])

    def get_prediction(self, x, name=''):
        if attrgetter(self.root, f'{name}.leaf_value') is not None:
            return attrgetter(self.root, f'{name}.leaf_value')[0]
        if name == '':   # TODO: This one is ugly
            if x[attrgetter(self.root, f'variable')] < attrgetter(self.root, f'threshold'):
                return self.get_prediction(x, name='left')
            else:
                return self.get_prediction(x, name='right')
        else:
            if x[attrgetter(self.root, f'{name}.variable')] < attrgetter(self.root, f'{name}.threshold'):
                return self.get_prediction(x, name=f'{name}.left')
            else:
                return self.get_prediction(x, name=f'{name}.right')

    def predict(self, new_data):
        return np.array([self.get_prediction(x) for x in new_data])


if __name__ == '__main__':
    iris = datasets.load_iris()
    tree = DecisionTree(data=iris.data, target=iris.target, max_depth=6)
    tree.fit()
    print(tree)
    print(tree.get_prediction(iris.data[0,:]))
    x = tree.predict(iris.data)
    print(np.unique(x, return_counts=True))

    # tree.find_best_split(iris.data, iris.target)
    #
    # print(tree.information_gain(np.array([0,0,0,1,1,1]),
    #                             np.array([0,0,0,1,1,1]),
    #                             np.array([0,0,0,1,1,1])))
    # print(tree.information_gain(np.array([0,0,0,1,1,1]),
    #                             np.array([0,0,0]),
    #                             np.array([1,1,1])))
    # print(tree.information_gain(np.array([0,0,0,1,1,1]),
    #                             np.array([0,1,0,1]),
    #                             np.array([0,1])))
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
