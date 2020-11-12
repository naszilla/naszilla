
import copy
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


def adj_mutate(nasbench,
               matrix,
               ops,
               cont,
               mutation_rate=1.0,
               patience=5000):
    """
    similar to perturb. A stochastic approach to perturbing the cell
    inspird by https://github.com/google-research/nasbench
    """
    p = 0
    while p < patience:
        p += 1
        new_matrix = copy.deepcopy(matrix)
        new_ops = copy.deepcopy(ops)

        if not cont:
            # flip each edge w.p. so expected flips is 1. same for ops
            edge_mutation_prob = mutation_rate / (NUM_VERTICES * (NUM_VERTICES - 1) / 2)
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if np.random.rand() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

        else:
            # add, remove, or change one edge
            diff = np.random.choice([-1, 0, 1])
            num_edges = np.array(matrix).sum()
            triu_indices = []
            for i in range(0, 6):
                for j in range(i + 1, 7):
                    triu_indices.append((i, j))

            if diff <= 0:
                # choose a random edge to remove
                idx = np.random.choice(range(num_edges))
                counter = 0
                for (i,j) in triu_indices:
                    if matrix[i][j] == 1:
                        if counter == idx:
                            new_matrix[i][j] = 0
                            break
                        else:
                            counter += 1
            if diff >= 0:
                # choose a random edge to add
                idx = np.random.choice(range(len(triu_indices) - num_edges))
                counter = 0
                for (i,j) in triu_indices:
                    if matrix[i][j] == 0:
                        if counter == idx:
                            new_matrix[i][j] = 1
                            break                                
                        else:
                            counter += 1

        op_mutation_prob = mutation_rate / OP_SPOTS
        for ind in range(1, OP_SPOTS + 1):
            if np.random.rand() < op_mutation_prob:
                available = [o for o in OPS if o != new_ops[ind]]
                new_ops[ind] = np.random.choice(available)

        new_spec = api.ModelSpec(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            return {
                'matrix': new_matrix,
                'ops': new_ops
            }
    return adj_mutate(nasbench, matrix, ops, mutation_rate=mutation_rate+1)


def trunc_adj_mutate(nasbench,
                     matrix,
                     ops,
                     cutoff,
                     mutation_rate=1.0,
                     patience=5000):

        p = 0
        while p < patience:
            p += 1
            new_matrix = copy.deepcopy(matrix)
            new_ops = copy.deepcopy(ops)

            trunc_op_spots = (max(cutoff, 21) - 21) // 2
            if cutoff >= 21 and ((cutoff % 2) == 0):
                if np.random.rand() > .5:
                    cutoff += 1

            if trunc_op_spots > 0:
                op_mutation_prob = mutation_rate / trunc_op_spots
                for ind in range(1, trunc_op_spots + 1):
                    if np.random.rand() < op_mutation_prob:
                        available = [o for o in OPS if o != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)

            trunc_edge_spots = max(cutoff, 21)
            if trunc_edge_spots > 0:
                edge_mutation_prob = mutation_rate / trunc_edge_spots
                # flip each edge w.p. so expected flips is 1. same for ops
                n = cutoff
                for src in range(0, NUM_VERTICES - 1):
                    if n <= 0:
                        break
                    for dst in range(src + 1, NUM_VERTICES):
                        n -= 1
                        if n <= 0:
                            break
                        if np.random.rand() < edge_mutation_prob:
                            new_matrix[src, dst] = 1 - new_matrix[src, dst]

            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops
                }
        return trunc_adj_mutate(nasbench, matrix, ops, cutoff, mutation_rate+1)


def path_mutate(nasbench,
                path_indices,
                index_hash,
                cont,
                weighted=True,
                cutoff=0,
                mutation_rate=1.0,
                patience=5000):

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

    p = 0
    while p < patience:
        p += 1
        new_path_indices = []
        n = 0

        if weighted:
            probs_by_length = [.2, .127, 3.36*10**-2, 3.92*10**-3, 1.5*10**-4, 6.37*10**-7]
        else:
            probs_by_length = [1/cutoff for i in range(6)]

        if not cont:
            # randomly sample paths
            for i in range(OP_SPOTS + 1):
                for j in range(len(OPS)**i):
                    prob = np.random.rand() * mutation_rate
                    if prob < probs_by_length[i] and n not in path_indices:
                        new_path_indices.append(n)
                    elif prob > probs_by_length[i] and n in path_indices:
                        new_path_indices.append(n)

                    n += 1
                    if n >= cutoff:
                        break
                if n >= cutoff:
                    break

            # add the paths after cutoff
            for path in path_indices:
                if path > cutoff:
                    new_path_indices.append(path)

        else:
            diff = np.random.choice([-1, 0, 1])
            new_path_indices = [path for path in path_indices]
            num_paths = len([i for i in path_indices if i < cutoff])

            # choose a random path to remove
            if diff <= 0:
                choices = [i for i in range(num_paths)]
                if len(choices) > 0:
                    idx = np.random.choice(choices)
                    new_path_indices.remove(path_indices[idx])
                else:
                    diff = 1

            # choose a random path to add
            if diff >= 0:
                choices = [i for i in range(cutoff) if i not in path_indices]
                if len(choices) > 0:
                    path = np.random.choice([i for i in range(cutoff) if i not in path_indices])
                    new_path_indices.append(path)
                elif len(new_path_indices) > 0:
                    new_path_indices.pop(len(new_path_indices)-1)

        new_path_indices.sort()
        new_path_indices = tuple(new_path_indices)

        if (new_path_indices is not None) and (new_path_indices in index_hash):
            spec = index_hash[new_path_indices]
            matrix = spec['matrix']
            ops = spec['ops']
            model_spec = api.ModelSpec(matrix, ops)

            if nasbench.is_valid(model_spec):
                return {
                    'matrix': matrix,
                    'ops': ops
                }
