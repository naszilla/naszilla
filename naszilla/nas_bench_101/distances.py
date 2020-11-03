
import numpy as np

CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]


def adj_distance(cell_1, cell_2):
    """
    compute the distance between two architectures
    by comparing their adjacency matrices and op lists
    (edit distance)
    """
    graph_dist = np.sum(np.array(cell_1.get_matrix()) != np.array(cell_2.get_matrix()))
    ops_dist = np.sum(np.array(cell_1.get_ops()) != np.array(cell_2.get_ops()))
    return graph_dist + ops_dist


def cont_adj_distance(cell_1, cell_2):
    """
    compute the distance between two architectures
    by comparing their adjacency matrices and op lists
    """
    graph_dist = np.sum(np.array(cell_1.get_matrix()) != np.array(cell_2.get_matrix()))
    ops_dist = np.sum(np.array(cell_1.get_ops()) != np.array(cell_2.get_ops()))
    delta_edges = np.abs(np.array(cell_1.get_matrix()).sum() - np.array(cell_2.get_matrix()).sum())
    return graph_dist + ops_dist + delta_edges


def path_distance(cell_1, cell_2, cutoff=None):
    """ 
    compute the distance between two architectures
    by comparing their path encodings
    """
    if cutoff:
        return np.sum(np.array(cell_1.encode('trunc_path', cutoff=cutoff) != np.array(cell_2.encode('trunc_path', cutoff=cutoff))))
    else:
        return np.sum(np.array(cell_1.encode('path') != np.array(cell_2.encode('path'))))


def cont_path_distance(cell_1, cell_2, cutoff=None):
    if cutoff:
        path_dist = np.sum(np.array(cell_1.encode('trunc_path', cutoff=cutoff)) != np.array(cell_2.encode('trunc_path', cutoff=cutoff)))
        delta_paths = np.abs(np.sum(cell_1.encode('trunc_path', cutoff=cutoff)) - np.sum(cell_2.encode('trunc_path', cutoff=cutoff)))
    else:
        path_dist = np.sum(np.array(cell_1.encode('path')) != np.array(cell_2.encode('path')))
        delta_paths = np.abs(np.sum(cell_1.encode('path')) - np.sum(cell_2.encode('path')))
    return path_dist + delta_paths


def nasbot_distance(cell_1, cell_2):
    # distance based on optimal transport between row sums, column sums, and ops

    cell_1_row_sums = sorted(np.array(cell_1.get_matrix()).sum(axis=0))
    cell_1_col_sums = sorted(np.array(cell_1.get_matrix()).sum(axis=1))

    cell_2_row_sums = sorted(np.array(cell_2.get_matrix()).sum(axis=0))
    cell_2_col_sums = sorted(np.array(cell_2.get_matrix()).sum(axis=1))

    row_dist = np.sum(np.abs(np.subtract(cell_1_row_sums, cell_2_row_sums)))
    col_dist = np.sum(np.abs(np.subtract(cell_1_col_sums, cell_2_col_sums)))

    cell_1_counts = [cell_1.get_ops().count(op) for op in OPS]
    cell_2_counts = [cell_2.get_ops().count(op) for op in OPS]

    ops_dist = np.sum(np.abs(np.subtract(cell_1_counts, cell_2_counts)))

    return row_dist + col_dist + ops_dist

