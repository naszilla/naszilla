import numpy as np
from nasbench import api

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


def random_cell_constrained(nasbench, max_edges=10, max_nodes=8):
    # get random cell with edges <= max_edges and nodes <= max_nodes
    while True:
        matrix = np.random.choice(
            [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        edges, nodes = Cell(matrix=matrix, ops=[]).num_edges_and_vertices()
        if edges <= max_edges and nodes <= max_nodes:
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if nasbench.is_valid(spec):
                return {
                    'matrix': matrix,
                    'ops': ops
                }   


def random_cell_uniform(cls, nasbench):
    # true uniform random
    hash_list = list(nasbench.hash_iterator())
    n = len(hash_list)
    num = np.random.randint(n)
    unique_hash = hash_list[num]
    fix, _ = nasbench.get_metrics_from_hash(unique_hash)
    cell = {'matrix':fix['module_adjacency'], 'ops':fix['module_operations']}
    return cls.convert_to_cell(cell)


def random_cell_adj(nasbench):
    """ 
    From the NASBench repository:
    one-hot adjacency matrix
    draw [0,1] for each slot in the adjacency matrix
    """
    while True:
        matrix = np.random.choice(
            [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return {
                'matrix': matrix,
                'ops': ops
            }


def random_cell_cont_adj(nasbench):
    """ 
    continuous adjacency matrix
    draw num_paths randomly
    draw continuous [0,1] for each edge, then threshold
    """
    while True:
        values = np.random.random(size=(NUM_VERTICES, NUM_VERTICES))
        values = np.triu(values, 1)
        n = np.random.randint(8) + 1
        flat = values.flatten()
        threshold = flat[np.argsort(flat)[-1 * n]]

        # now convert it to a model spec
        matrix = np.random.choice([0], size=(NUM_VERTICES, NUM_VERTICES))
        for i in range(NUM_VERTICES):
            for j in range(NUM_VERTICES):
                if values[i][j] >= threshold:
                    matrix[i][j] = 1

        ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
        ops[0] = INPUT
        ops[-1] = OUTPUT

        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            return {
                'matrix': matrix,
                'ops': ops
            }


def random_cell_path(nasbench, index_hash, weighted, cont, cutoff):
    """ 
    continuous path encoding:
    draw num_paths randomly
    draw continuous [0,1]*weight for each path, then threshold
    """

    """
    For NAS encodings experiments, some of the path-based encodings currently require a
    hash map from path indices to cell architectuers. We have created a pickle file which
    contains the hash map, located at 
    https://drive.google.com/file/d/1yMRFxT6u3ZyfiWUPhtQ_B9FbuGN3X-Nf/view?usp=sharing
    """
    if not index_hash:
        print('Error: please download index_hash, located at \
        https://drive.google.com/file/d/1yMRFxT6u3ZyfiWUPhtQ_B9FbuGN3X-Nf/view?usp=sharing')
        raise NotImplementedError()

    total_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
    if not cutoff:
        cutoff = total_paths

    while True:
        if weighted:
            probs_by_length = [.2, .127, 3.36*10**-2, 3.92*10**-3, 1.5*10**-4, 6.37*10**-7]
        else:
            probs_by_length = [1/cutoff for i in range(OP_SPOTS + 1)]

        # give continuous value to all paths
        path_probs = []
        weights = []
        n = 0            
        for i in range(OP_SPOTS + 1):
            for j in range(len(OPS)**i):
                weights.append(probs_by_length[i])
                path_probs.append(np.random.rand())
                n += 1
                if n >= cutoff:
                    break
            if n >= cutoff:
                break

        if cont:
            # threshold to get num_paths best paths
            num_paths = np.random.choice([i for i in range(1, 7)])
            weighted_probs = [path_probs[i] * weights[i] for i in range(len(path_probs))]
            path_indices = np.argsort(weighted_probs)[-1 * num_paths:]
        else:
            # pick each path with some probability
            path_indices = [i for i in range(len(path_probs)) if path_probs[i] < weights[i]]

        # convert to a model spec
        path_indices.sort()
        path_indices = tuple(path_indices)
        if path_indices in index_hash:
            spec = index_hash[path_indices]
            matrix = spec['matrix']
            ops = spec['ops']
            model_spec = api.ModelSpec(matrix, ops)

            if nasbench.is_valid(model_spec):
                return {
                    'matrix': matrix,
                    'ops': ops
                }
