import numpy as np
import copy
import itertools
import random
import sys
import pickle
from collections import namedtuple

import nasbench301 as nb

OPS = ['max_pool_3x3',
       'avg_pool_3x3',
       'skip_connect',
       'sep_conv_3x3',
       'sep_conv_5x5',
       'dil_conv_3x3',
       'dil_conv_5x5'
       ]
NUM_VERTICES = 4
INPUT_1 = 'c_k-2'
INPUT_2 = 'c_k-1'
OUTPUT = 'c_k'


class Cell301:

    def __init__(self, arch):
        self.arch = arch

    def serialize(self):
        return tuple([tuple([tuple(pair) for pair in cell]) for cell in self.arch])

    def get_val_loss(self, nasbench, deterministic=True, patience=50, epochs=None, dataset=None):

        genotype = self.convert_to_genotype(self.arch)
        acc = nasbench[0].predict(config=genotype, representation="genotype")
        return 100 - acc
        
    def get_test_loss(self, nasbench, patience=50, epochs=None, dataset=None):
        # currently only val_loss is supported. Just return the val loss here
        return self.get_val_loss(nasbench)
        
    def convert_to_genotype(self, arch):

        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        op_dict = {
        0: 'max_pool_3x3',
        1: 'avg_pool_3x3',
        2: 'skip_connect',
        3: 'sep_conv_3x3',
        4: 'sep_conv_5x5',
        5: 'dil_conv_3x3',
        6: 'dil_conv_5x5'
        }

        darts_arch = [[], []]
        i=0
        for cell in arch:
            for n in cell:
                darts_arch[i].append((op_dict[n[1]], n[0]))
            i += 1
        genotype = Genotype(normal=darts_arch[0], normal_concat=[2,3,4,5], reduce=darts_arch[1], reduce_concat=[2,3,4,5])
        return genotype

    def make_mutable(self):
        # convert tuple to list so that it is mutable
        arch_list = []
        for cell in self.arch:
            arch_list.append([])
            for pair in cell:
                arch_list[-1].append([])
                for num in pair:
                    arch_list[-1][-1].append(num)
        return arch_list
    
    def encode(self, predictor_encoding, nasbench=None, deterministic=True, cutoff=None):

        if predictor_encoding == 'path':
            return self.encode_paths()
        elif predictor_encoding == 'trunc_path':
            if not cutoff:
                cutoff = 100
            return self.encode_paths(cutoff=cutoff)
        elif predictor_encoding == 'adj':
            return self.encode_adj()

        else:
            print('{} is not yet implemented as a predictor encoding \
             for nasbench301'.format(predictor_encoding))
            raise NotImplementedError()

    def distance(self, other, dist_type, cutoff=None):

        if dist_type == 'path':
            return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))
        elif dist_type == 'adj':
            return np.sum(np.array(self.encode_adj() != np.array(other.encode_adj())))
        else:
            print('{} is not yet implemented as a distance for nasbench301'.format(dist_type))
            raise NotImplementedError()

    def mutate(self, 
               nasbench,
               mutation_rate=1,
               mutate_encoding='adj',
               cutoff=None,
               comparisons=2500,
               patience=5000,
               prob_wt=False,
               index_hash=None):

        if mutate_encoding != 'adj':
            print('{} is not yet implemented as a mutation \
                encoding for nasbench301'.format(mutate_encoding))
            raise NotImplementedError()

        """ mutate a single arch """
        # first convert tuple to array so that it is mutable
        mutation = self.make_mutable()

        #make mutations
        for _ in range(int(mutation_rate)):
            cell = np.random.choice(2)
            pair = np.random.choice(len(OPS))
            num = np.random.choice(2)
            if num == 1:
                mutation[cell][pair][num] = np.random.choice(len(OPS))
            else:
                inputs = pair // 2 + 2
                choice = np.random.choice(inputs)
                if pair % 2 == 0 and mutation[cell][pair+1][num] != choice:
                    mutation[cell][pair][num] = choice
                elif pair % 2 != 0 and mutation[cell][pair-1][num] != choice:
                    mutation[cell][pair][num] = choice
                      
        return {'arch': mutation}

    @classmethod
    def random_cell(cls, 
                    nasbench, 
                    random_encoding, 
                    cutoff=None,
                    max_edges=10, 
                    max_nodes=8,
                    index_hash=None):
        # output a uniformly random architecture spec
        # from the DARTS repository
        # https://github.com/quark0/darts

        if random_encoding != 'adj':
            print('{} is not yet implemented as a mutation \
                encoding for nasbench301'.format(random_encoding))
            raise NotImplementedError()

        normal = []
        reduction = []
        for i in range(NUM_VERTICES):
            ops = np.random.choice(range(len(OPS)), NUM_VERTICES)

            #input nodes for conv
            nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
            #input nodes for reduce
            nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)

            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])

        return {'arch': (normal, reduction)}

    def perturb(self, nasbench, edits=1):
        return self.mutate()

    def get_paths(self):
        """ return all paths from input to output """

        path_builder = [[[], [], [], []], [[], [], [], []]]
        paths = [[], []]

        for i, cell in enumerate(self.arch):
            for j in range(8):
                if cell[j][0] == 0:
                    path = [INPUT_1, OPS[cell[j][1]]]
                    path_builder[i][j//2].append(path)
                    paths[i].append(path)
                elif cell[j][0] == 1:
                    path = [INPUT_2, OPS[cell[j][1]]]
                    path_builder[i][j//2].append(path)
                    paths[i].append(path)
                else:
                    for path in path_builder[i][cell[j][0] - 2]:
                        path = [*path, OPS[cell[j][1]]]
                        path_builder[i][j//2].append(path)
                        paths[i].append(path)

        return paths

    def get_path_indices(self, long_paths=True):
        """
        compute the index of each path
        There are 4 * (8^0 + ... + 8^4) paths total
        If long_paths = False, we give a single boolean to all paths of
        size 4, so there are only 4 * (1 + 8^0 + ... + 8^3) paths
        """
        paths = self.get_paths()
        normal_paths, reduce_paths = paths
        num_ops = len(OPS)
        """
        Compute the max number of paths per input per cell.
        Since there are two cells and two inputs per cell, 
        total paths = 4 * max_paths
        """

        max_paths = sum([num_ops ** i for i in range(NUM_VERTICES + 1)])    
        path_indices = []

        # set the base index based on the cell and the input
        for i, paths in enumerate((normal_paths, reduce_paths)):
            for path in paths:
                index = i * 2 * max_paths
                if path[0] == INPUT_2:
                    index += max_paths

                # recursively compute the index of the path
                for j in range(NUM_VERTICES + 1):
                    if j == len(path) - 1:
                        path_indices.append(index)
                        break
                    elif j == (NUM_VERTICES - 1) and not long_paths:
                        path_indices.append(2 * (i + 1) * max_paths - 1)
                        break
                    else:
                        index += num_ops ** j * (OPS.index(path[j + 1]) + 1)

        return tuple(path_indices)

    def encode_paths(self, cutoff=None):
        # output one-hot encoding of paths
        path_indices = self.get_path_indices()
        num_ops = len(OPS)

        max_paths = sum([num_ops ** i for i in range(NUM_VERTICES + 1)])    

        path_encoding = np.zeros(4 * max_paths)
        for index in path_indices:
            path_encoding[index] = 1
        if cutoff:
            path_encoding = path_encoding[:cutoff]
        return path_encoding

    def encode_adj(self):
        matrices = []
        ops = []
        true_num_vertices = NUM_VERTICES + 3
        for cell in self.arch:
            matrix = np.zeros((true_num_vertices, true_num_vertices))
            op_list = []
            for i, edge in enumerate(cell):
                dest = i//2 + 2
                matrix[edge[0]][dest] = 1
                op_list.append(edge[1])
            for i in range(2, 6):
                matrix[i][-1] = 1
            matrices.append(matrix)
            ops.append(op_list)

        encoding = [*matrices[0].flatten(), *ops[0], *matrices[1].flatten(), *ops[1]]
        return np.array(encoding)

    def get_neighborhood(self,
                         nasbench, 
                         mutate_encoding='adj',
                         cutoff=None,
                         index_hash=None, 
                         shuffle=True):
        if mutate_encoding != 'adj':
            print('{} is not yet implemented as a neighborhood for nasbench301'.format(mutate_encoding))
            raise NotImplementedError()

        op_nbhd = []
        edge_nbhd = []

        for i, cell in enumerate(self.arch):
            for j, pair in enumerate(cell):

                # mutate the op
                available = [op for op in range(len(OPS)) if op != pair[1]]
                for op in available:
                    new_arch = self.make_mutable()
                    new_arch[i][j][1] = op
                    op_nbhd.append({'arch': new_arch})

                # mutate the edge
                other = j + 1 - 2 * (j % 2)
                available = [edge for edge in range(j//2+2) \
                             if edge not in [cell[other][0], pair[0]]] 

                for edge in available:
                    new_arch = self.make_mutable()
                    new_arch[i][j][0] = edge
                    edge_nbhd.append({'arch': new_arch})

        if shuffle:
            random.shuffle(edge_nbhd)
            random.shuffle(op_nbhd)

        # 112 in edge nbhd, 24 in op nbhd
        # alternate one edge nbr per 4 op nbrs
        nbrs = []
        op_idx = 0
        for i in range(len(edge_nbhd)):
            nbrs.append(edge_nbhd[i])
            for j in range(4):
                nbrs.append(op_nbhd[op_idx])
                op_idx += 1
        nbrs = [*nbrs, *op_nbhd[op_idx:]]

        return nbrs

    def get_num_params(self, nasbench):
        # todo: add this method
        return 100
