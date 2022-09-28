from sklearn.tree import DecisionTreeRegressor, export_graphviz
from functools import partial
import numpy as np
import typing
import re

# transform y
def y_inverse_transform(y: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Return logarithmic kinetics data to the original domain."""
    y = y*std+mean
    y = np.exp(y)

    return y

def y_transform(y: np.ndarray) -> tuple[np.ndarray, typing.Callable]:
    """Transforms kinetics data for easier fitting.

    The natural logarithm is applied, and the data is transformed to have zero mean and unit variance.
    Returns the transformed data and the necessary function for the reverse transform."""
    y = np.log(y)
    mean = np.mean(y)
    std = np.std(y)
    y = (y-mean)/std
    inverse_function = partial(y_inverse_transform, mean=mean, std=std)

    return y, inverse_function, mean, std

def y_transform_fixed(y, mean, std):
    """Transforms kinetics data based on given mean and std for fitting of test data."""
    y = np.log(y)
    y = (y-mean)/std

    return y

# tree analysis
def analyze_tree(my_tree: DecisionTreeRegressor, features: list[str]) -> tuple[int, np.ndarray]:
    """Analyzes a given decision tree. 
    
    Returns the number of leaves in a tree and how often each feature was used for a decisions."""
    n_nodes = my_tree.tree_.node_count
    children_left = my_tree.tree_.children_left
    children_right = my_tree.tree_.children_right
    feature = my_tree.tree_.feature
    threshold = my_tree.tree_.threshold

    # label all nodes
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # pop ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to stack so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    # add occurences of features in selection
    leaves = 0
    occurrences = np.zeros(len(features))
    for i in range(n_nodes):
        if is_leaves[i]:
            leaves += 1
        else:
            occurrences[feature[i]] += 1

    return leaves, occurrences

def visualize_tree(my_tree: DecisionTreeRegressor, columns: list[str], y_inversion_func) -> str:
    """Exports a graphviz dot file of a regression tree for visualization."""
    text = export_graphviz(my_tree, out_file=None, feature_names=columns)

    # delete n samples and squared error
    indices = [(m.start(), m.end()) for m in re.finditer(r'\\nsquared_error = [0-9\.]*\\nsamples = [0-9\.]*', text)]
    temp_text = text
    for index in indices:
        temp_text = temp_text.replace(text[index[0]:index[1]],'')
    text = temp_text

    indices = [(m.start(), m.end()) for m in re.finditer(r'squared_error = [0-9\.]*\\nsamples = [0-9\.]*\\n', text)]
    temp_text = text
    for index in indices:
        temp_text = temp_text.replace(text[index[0]:index[1]],'')
    text = temp_text

    # replace numbers
    indices = [(m.start(1), m.end(1)) for m in re.finditer(r"value = (-?[0-9\.]*)", text)]
    temp_text = text
    for index in indices:
        number = float(text[index[0]:index[1]])
        inv_number = float(y_inversion_func(number))
        temp_text = temp_text.replace(str(number),"{:.2E}".format(inv_number))
    text = temp_text

    # delete (ensemble)
    text = text.replace(" (ensemble)","")
    # delete value =
    text = text.replace("value = ", "")

    return text