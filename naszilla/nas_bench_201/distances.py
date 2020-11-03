
import numpy as np

OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
NUM_OPS = len(OPS)
OP_SPOTS = 6
LONGEST_PATH_LENGTH = 3


def adj_distance(cell_1, cell_2):
    """
    compute the distance between two architectures
    by comparing their adjacency matrices and op lists
    (edit distance)
    """
    graph_dist = np.sum(np.array(cell_1.get_matrix()) != np.array(cell_2.get_matrix()))
    ops_dist = np.sum(np.array(cell_1.get_ops()) != np.array(cell_2.get_ops()))
    return graph_dist + ops_dist


def path_distance(cell_1, cell_2, cutoff=None):
    """ 
    compute the distance between two architectures
    by comparing their path encodings
    """
    if cutoff:
        return np.sum(np.array(cell_1.encode('trunc_path', cutoff=cutoff) != np.array(cell_2.encode('trunc_path', cutoff=cutoff))))
    else:
        return np.sum(np.array(cell_1.encode('path') != np.array(cell_2.encode('path'))))

def adj_distance(cell_1, cell_2):

    cell_1_ops = cell_1.get_op_list()
    cell_2_ops = cell_2.get_op_list()
    return np.sum([1 for i in range(len(cell_1_ops)) if cell_1_ops[i] != cell_2_ops[i]])

def nasbot_distance(cell_1, cell_2):
    # distance based on optimal transport between row sums, column sums, and ops

    cell_1_ops = cell_1.get_op_list()
    cell_2_ops = cell_2.get_op_list()

    cell_1_counts = [cell_1_ops.count(op) for op in OPS]
    cell_2_counts = [cell_2_ops.count(op) for op in OPS]
    ops_dist = np.sum(np.abs(np.subtract(cell_1_counts, cell_2_counts)))

    return ops_dist + adj_distance(cell_1, cell_2)