import numpy as np
import copy
import itertools
import random

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


class Cell:

    def __init__(self, matrix, ops):

        self.matrix = matrix
        self.ops = ops

    def serialize(self):
        return {
            'matrix': self.matrix,
            'ops': self.ops
        }

    def modelspec(self):
        return api.ModelSpec(matrix=self.matrix, ops=self.ops)

    @classmethod
    def random_cell(cls, nasbench):
        """ from the NASBench repository """
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

    def get_val_loss(self, nasbench, deterministic=1, patience=50):
        if not deterministic:
            # output one of the three validation accuracies at random
            return (100*(1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']))
        else:        
            # query the api until we see all three accuracies, then average them
            # note: a few architectures only have two accuracies, so we need a while loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100*(1-np.mean(accs)), 3)            


    def get_test_loss(self, nasbench, patience=50):
        """
        query the api until we see all three accuracies, then average them
        note: a few architectures only have two accuracies
        """
        accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['test_accuracy']
            if acc not in accs:
                accs.append(acc)
        return round(100*(1-np.mean(accs)), 3)

    def perturb(self, nasbench, edits=1):
        """ create new perturbed cell """
        new_matrix = copy.deepcopy(self.matrix)
        new_ops = copy.deepcopy(self.ops)
        for _ in range(edits):
            while True:
                if np.random.random() < 0.5:
                    for src in range(0, NUM_VERTICES - 1):
                        for dst in range(src+1, NUM_VERTICES):
                            new_matrix[src][dst] = 1 - new_matrix[src][dst]
                else:
                    for ind in range(1, NUM_VERTICES - 1):
                        available = [op for op in OPS if op != new_ops[ind]]
                        new_ops[ind] = np.random.choice(available)

                new_spec = api.ModelSpec(new_matrix, new_ops)
                if nasbench.is_valid(new_spec):
                    break
        return {
            'matrix': new_matrix,
            'ops': new_ops
        }

    def mutate(self, nasbench, mutation_rate=1.0):
        """similar to perturb. A stochastic approach to perturbing the cell"""
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            edge_mutation_prob = mutation_rate / NUM_VERTICES
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutation_rate / OP_SPOTS
            for ind in range(1, OP_SPOTS + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in OPS if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return {
                    'matrix': new_matrix,
                    'ops': new_ops
                }

    def encode_cell(self):
        """ adjacency matrix + op list encoding """
        encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS
        encoding = np.zeros((encoding_length))
        dic = {CONV1X1: 0., CONV3X3: 0.5, MAXPOOL3X3: 1.0}
        n = 0
        for i in range(NUM_VERTICES - 1):
            for j in range(i+1, NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, NUM_VERTICES - 1):
            encoding[-i] = dic[self.ops[i]]
        return tuple(encoding)

    def get_paths(self):
        """ return all paths from input to output"""
        paths = []
        for j in range(0, NUM_VERTICES):
            paths.append([[]]) if self.matrix[0][j] else paths.append([])
        # create paths sequentially
        for i in range(1, NUM_VERTICES - 1):
            for j in range(1, NUM_VERTICES):
                if self.matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, self.ops[i]])
        return paths[-1]

    def get_path_indices(self):
        """
        convert paths from arrays to ints
        There are 3^0 + ... + 3^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are three choices for the operation.)
        """
        paths = self.get_paths()
        mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
        path_indices = []

        for path in paths:
            index = 0
            for i in range(NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(OPS) ** i * (mapping[path[i]] + 1)

        return tuple(path_indices)

    def encode_paths(self):
        """ output one-hot encoding of paths """
        num_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])
        path_indices = self.get_path_indices()
        path_encoding = np.zeros(num_paths)
        for index in path_indices:
            path_encoding[index] = 1
        return path_encoding

    def path_distance(self, other):
        return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))

    def edit_distance(self, other):
        graph_dist = np.sum(np.array(self.matrix) != np.array(other.matrix))
        ops_dist = np.sum(np.array(self.ops) != np.array(other.ops))
        return (graph_dist + ops_dist)

    # not currently being used. Remove in release version
    def perm_distance(self, other, perm):
        """ helper function to nasbot_distance """
        op_distance = 0
        for i in range(1, 6):
            if self.ops[i] != other.ops[perm[i - 1]]:
                if self.ops[i][0] == 'c' and other.ops[perm[i - 1]][0] == 'c':
                    op_distance += 1
                else:
                    op_distance += 2

        edge_distance = 0
        for i in range(1, 5):
            for j in range(1, 5):
                if self.matrix[i][j] != other.matrix[perm[i - 1]][perm[j - 1]]:
                    op_distance += 1

        return op_distance + edge_distance

    def nasbot_distance(self, other):
        distance = 1000
        for perm in itertools.combinations((1, 2, 3, 4, 5)):
            perm_distance = self.perm_distance(other, perm)
            if perm_distance < distance:
                distance = perm_distance

        distance /= 10.0
        return distance

    # the next few methods are helper functions for nasbot_path_distance
    def path_edit_dist(self, path_1, path_2):
        dist = 0
        for i, op in enumerate(path_1):
            up, down, op_dist = i, i, 0
            while op not in (path_2[up], path_2[down]):
                up = min(up+1, len(path_2)-1)
                down = max(down-1, 0)
                op_dist += 1
            dist += op_dist
        return dist

    def path_symmetric_diff(self, path, op_diffs, mapping):
        path_diff = []
        for i, op in enumerate(path):
            if op_diffs[mapping[op]] > 0:
                op_diffs[mapping[op]] -= 1
            else:
                path_diff.append(op)
        return path_diff

    def path_dist(self, path_1, path_2):
        #get counts of the ops
        mapping = {'conv3x3-bn-relu':0, 'conv1x1-bn-relu':1, 'maxpool3x3':2}
        op_counts = np.zeros((2, 3))
        for i, path in enumerate((path_1, path_2)):
            for op in path:
                op_counts[i][mapping[op]] += 1

        op_difference_count = sum(abs(op_counts[1] - op_counts[0]))

        path_diff_1 = self.path_symmetric_diff(path_1, op_counts[0] - op_counts[1], mapping)
        path_diff_2 = self.path_symmetric_diff(path_2, op_counts[1] - op_counts[0], mapping)

        distance = 1 * abs(len(path_1) - len(path_2)) + 5 * op_difference_count + 3 * self.path_edit_dist(path_diff_1, path_diff_2)
        distance /= np.log2(len(path_1) + len(path_2) + 3)

        return distance

    def nasbot_path_distance(self, other):

        paths_1 = self.get_paths()
        paths_2 = other.get_paths()

        if len(paths_2) < len(paths_1):
            paths_1, paths_2 = paths_2, paths_1

        min_dist = -1
        for mapping in itertools.permutations(range(len(paths_2)), len(paths_1)):
            dist = 0
            for i in range(len(paths_1)):
                dist += self.path_dist(paths_1[i], paths_2[mapping[i]]) / np.log2(len(paths_1)+len(paths_2))
            if dist < min_dist or min_dist == -1:
                min_dist = dist

        distance = 1 * min_dist + 2 * abs(len(paths_1) - len(paths_2))
        distance = distance / 2
        return distance


