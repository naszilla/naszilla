import numpy as np
import sys
import os
import copy
import random

sys.path.append(os.path.expanduser('~/darts/cnn'))
from train_class import Train

OPS = ['none',
       'max_pool_3x3',
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


class Arch:

    def __init__(self, arch):
        self.arch = arch

    def serialize(self):
        return self.arch

    def query(self, epochs=50):
        trainer = Train()
        val_losses, test_losses = trainer.main(self.arch, epochs=epochs)
        val_loss = 100 - np.mean(val_losses)
        test_loss = 100 - test_losses[-1]        
        return val_loss, test_loss

    @classmethod
    def random_arch(cls):
        # output a uniformly random architecture spec
        # from the DARTS repository
        # https://github.com/quark0/darts

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

        return (normal, reduction)

    def get_arch_list(self):
        # convert tuple to list so that it is mutable
        arch_list = []
        for cell in self.arch:
            arch_list.append([])
            for pair in cell:
                arch_list[-1].append([])
                for num in pair:
                    arch_list[-1][-1].append(num)
        return arch_list

    def mutate(self, edits):
        """ mutate a single arch """
        # first convert tuple to array so that it is mutable
        mutation = self.get_arch_list()

        #make mutations
        for _ in range(edits):
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
                      
        return mutation

    def get_paths(self):
        """ return all paths from input to output """

        path_builder = [[[], [], [], []], [[], [], [], []]]
        paths = [[], []]

        for i, cell in enumerate(self.arch):
            for j in range(len(OPS)):
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

        # check if there are paths of length >=5
        contains_long_path = [False, False]
        if max([len(path) for path in paths[0]]) >= 5:
            contains_long_path[0] = True
        if max([len(path) for path in paths[1]]) >= 5:
            contains_long_path[1] = True

        return paths, contains_long_path

    def get_path_indices(self, long_paths=True):
        """
        compute the index of each path
        There are 4 * (8^0 + ... + 8^4) paths total
        If long_paths = False, we give a single boolean to all paths of
        size 4, so there are only 4 * (1 + 8^0 + ... + 8^3) paths
        """
        paths, contains_long_path = self.get_paths()
        normal_paths, reduce_paths = paths
        num_ops = len(OPS)
        """
        Compute the max number of paths per input per cell.
        Since there are two cells and two inputs per cell, 
        total paths = 4 * max_paths
        """
        if not long_paths:
            max_paths = 1 + sum([num_ops ** i for i in range(NUM_VERTICES)])
        else:
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

        return (tuple(path_indices), contains_long_path)

    def encode_paths(self, long_paths=True):
        # output one-hot encoding of paths
        path_indices, _ = self.get_path_indices(long_paths=long_paths)
        num_ops = len(OPS)

        if not long_paths:
            max_paths = 1 + sum([num_ops ** i for i in range(NUM_VERTICES)])
        else:
            max_paths = sum([num_ops ** i for i in range(NUM_VERTICES + 1)])    

        path_encoding = np.zeros(4 * max_paths)
        for index in path_indices:
            path_encoding[index] = 1
        return path_encoding

    def path_distance(self, other):
        # compute the distance between two architectures
        # by comparing their path encodings
        return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))





