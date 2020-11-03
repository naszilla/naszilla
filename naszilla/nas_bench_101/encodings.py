import numpy as np

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


def encode_adj(matrix, ops):
    """ 
    compute the "standard" encoding,
    i.e. adjacency matrix + op list encoding 
    """
    encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS
    encoding = np.zeros((encoding_length))
    dic = {CONV1X1: 0., CONV3X3: 0.5, MAXPOOL3X3: 1.0}
    n = 0
    for i in range(NUM_VERTICES - 1):
        for j in range(i+1, NUM_VERTICES):
            encoding[n] = matrix[i][j]
            n += 1
    for i in range(1, NUM_VERTICES - 1):
        encoding[-i] = dic[ops[i]]
    return tuple(encoding)


def encode_cat_adj(matrix, ops):
    encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS
    encoding = np.zeros((encoding_length))
    dic = {CONV1X1: 0., CONV3X3: 0.5, MAXPOOL3X3: 1.0}
    n = 0
    m = 0
    for i in range(NUM_VERTICES - 1):
        for j in range(i+1, NUM_VERTICES):
            if matrix[i][j]:
                encoding[m] = n
                m += 1
            n += 1

    for i in range(1, NUM_VERTICES - 1):
        encoding[-i] = dic[ops[i]]
    return tuple(encoding)


def encode_cont_adj(matrix, ops):
    """ 
    compute the continuous encoding from nasbench,
    num in [1,9], adjacency matrix with values in [0,1], and op list
    the edges are the num largest edges in the adjacency matrix
    """
    encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS + 1
    encoding = np.zeros((encoding_length))
    dic = {CONV1X1: 0., CONV3X3: 0.5, MAXPOOL3X3: 1.0}
    n = 0
    for i in range(NUM_VERTICES - 1):
        for j in range(i+1, NUM_VERTICES):
            encoding[n] = matrix[i][j]
            n += 1
    for i in range(1, NUM_VERTICES - 1):
        encoding[-i] = dic[ops[i]]
    encoding[-1] = matrix.sum()
    return tuple(encoding)


def encode_paths(path_indices):
    """ output one-hot encoding of paths """
    num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
    encoding = np.zeros(num_paths)
    for index in path_indices:
        encoding[index] = 1
    return encoding
