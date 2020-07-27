import numpy as np
import copy
import itertools
import random
import sys
import os
import pickle


OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
NUM_OPS = len(OPS)
OP_SPOTS = 6
LONGEST_PATH_LENGTH = 3

class Cell:

    def __init__(self, string):
        self.string = string

    def get_string(self):
        return self.string

    def serialize(self):
        return {
            'string':self.string
        }

    @classmethod
    def random_cell(cls, nasbench, max_nodes=4):
        """
        From the AutoDL-Projects repository
        """
        ops = []
        for i in range(OP_SPOTS):
            op = random.choice(OPS)
            ops.append(op)
        return {'string':cls.get_string_from_ops(ops)}


    def get_runtime(self, nasbench, dataset='cifar100'):
        return nasbench.query_by_index(index, dataset).get_eval('x-valid')['time']

    def get_val_loss(self, nasbench, deterministic=1, dataset='cifar100'):
        index = nasbench.query_index_by_arch(self.string)
        if dataset == 'cifar10':
            results = nasbench.query_by_index(index, 'cifar10-valid')
        else:
            results = nasbench.query_by_index(index, dataset)

        accs = []
        for key in results.keys():
            accs.append(results[key].get_eval('x-valid')['accuracy'])

        if deterministic:
            return round(100-np.mean(accs), 10)   
        else:
            return round(100-np.random.choice(accs), 10)

    def get_test_loss(self, nasbench, dataset='cifar100', deterministic=1):
        index = nasbench.query_index_by_arch(self.string)
        results = nasbench.query_by_index(index, dataset)

        accs = []
        for key in results.keys():
            accs.append(results[key].get_eval('ori-test')['accuracy'])

        if deterministic:
            return round(100-np.mean(accs), 4)   
        else:
            return round(100-np.random.choice(accs), 4)

    def get_op_list(self):
        # given a string, get the list of operations
        tokens = self.string.split('|')
        ops = [t.split('~')[0] for i,t in enumerate(tokens) if i not in [0,2,5,9]]
        return ops

    def get_num(self):
        # compute the unique number of the architecture, in [0, 15624]
        ops = self.get_op_list()
        index = 0
        for i, op in enumerate(ops):
            index += OPS.index(op) * NUM_OPS ** i
        return index

    @classmethod
    def get_string_from_ops(cls, ops):
        # given a list of operations, get the string
        strings = ['|']
        nodes = [0, 0, 1, 0, 1, 2]
        for i, op in enumerate(ops):
            strings.append(op+'~{}|'.format(nodes[i]))
            if i < len(nodes) - 1 and nodes[i+1] == 0:
                strings.append('+|')
        return ''.join(strings)

    def perturb(self, nasbench,
                mutation_rate=1):
        # more deterministic version of mutate
        ops = self.get_op_list()
        new_ops = []
        num = np.random.choice(len(ops))
        for i, op in enumerate(ops):
            if i == num:
                available = [o for o in OPS if o != op]
                new_ops.append(np.random.choice(available))
            else:
                new_ops.append(op)
        return {'string':self.get_string_from_ops(new_ops)}

    def mutate(self, 
               nasbench, 
               mutation_rate=1.0, 
               patience=5000):

        p = 0
        ops = self.get_op_list()
        new_ops = []
        # keeping mutation_prob consistent with nasbench_101
        mutation_prob = mutation_rate / (OP_SPOTS - 2)

        for i, op in enumerate(ops):
            if random.random() < mutation_prob:
                available = [o for o in OPS if o != op]
                new_ops.append(random.choice(available))
            else:
                new_ops.append(op)

        return {'string':self.get_string_from_ops(new_ops)}

    def encode_standard(self):
        """ 
        compute the standard encoding
        """
        ops = self.get_op_list()
        encoding = []
        for op in ops:
            encoding.append(OPS.index(op))

        return encoding

    def get_num_params(self, nasbench):
        # todo update to the newer nasbench-201 dataset
        return 100

    def get_paths(self):
        """ 
        return all paths from input to output
        """
        path_blueprints = [[3], [0,4], [1,5], [0,2,5]]
        ops = self.get_op_list()
        paths = []
        for blueprint in path_blueprints:
            paths.append([ops[node] for node in blueprint])

        return paths

    def get_path_indices(self):
        """
        compute the index of each path
        """
        paths = self.get_paths()
        path_indices = []

        for i, path in enumerate(paths):
            if i == 0:
                index = 0
            elif i in [1, 2]:
                index = NUM_OPS
            else:
                index = NUM_OPS + NUM_OPS ** 2
            for j, op in enumerate(path):
                index += OPS.index(op) * NUM_OPS ** j
            path_indices.append(index)

        return tuple(path_indices)

    def encode_paths(self):
        """ output one-hot encoding of paths """
        num_paths = sum([NUM_OPS ** i for i in range(1, LONGEST_PATH_LENGTH + 1)])
        path_indices = self.get_path_indices()
        encoding = np.zeros(num_paths)
        for index in path_indices:
            encoding[index] = 1
        return encoding

    def path_distance(self, other):
        """ 
        compute the distance between two architectures
        by comparing their path encodings
        """
        return np.sum(np.array(self.encode_paths() != np.array(other.encode_paths())))

    def trunc_path_distance(self, other, cutoff=30):
        """ 
        compute the distance between two architectures
        by comparing their truncated path encodings
        """
        paths = np.array(self.encode_paths()[cutoff])
        other_paths = np.array(other.encode_paths()[cutoff])
        return np.sum(paths != other_paths)

    def edit_distance(self, other):

        ops = self.get_op_list()
        other_ops = other.get_op_list()
        return np.sum([1 for i in range(len(ops)) if ops[i] != other_ops[i]])

    def nasbot_distance(self, other):
        # distance based on optimal transport between row sums, column sums, and ops

        ops = self.get_op_list()
        other_ops = other.get_op_list()

        counts = [ops.count(op) for op in OPS]
        other_counts = [other_ops.count(op) for op in OPS]
        ops_dist = np.sum(np.abs(np.subtract(counts, other_counts)))

        return ops_dist + self.edit_distance(other)
