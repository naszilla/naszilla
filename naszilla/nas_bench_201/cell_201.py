import numpy as np
import copy
import itertools
import random
import sys
import os
import pickle

from naszilla.nas_bench_201.distances import *

INPUT = 'input'
OUTPUT = 'output'
OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
NUM_OPS = len(OPS)
OP_SPOTS = 6
LONGEST_PATH_LENGTH = 3

class Cell201:

    def __init__(self, string):
        self.string = string

    def get_string(self):
        return self.string

    def serialize(self):
        return {
            'string':self.string
        }

    @classmethod
    def random_cell(cls, 
                    nasbench, 
                    max_nodes=None, 
                    max_edges=None,
                    cutoff=None,
                    index_hash=None,
                    random_encoding=None):
        """
        From the AutoDL-Projects repository
        """
        ops = []
        for i in range(OP_SPOTS):
            op = random.choice(OPS)
            ops.append(op)
        return {'string':cls.get_string_from_ops(ops)}

    def encode(self, predictor_encoding, nasbench=None, deterministic=True, cutoff=None):

        if predictor_encoding == 'adj':
            return self.encode_standard()
        elif predictor_encoding == 'path':
            return self.encode_paths()
        elif predictor_encoding == 'trunc_path':
            if not cutoff:
                cutoff = 30
            return self.encode_freq_paths(cutoff=cutoff)
        elif predictor_encoding == 'gcn':
            return self.gcn_encoding(nasbench, deterministic=deterministic)
        else:
            print('{} is an invalid predictor encoding'.format(predictor_encoding))
            raise NotImplementedError()

    def gcn_encoding(self, nasbench, deterministic):

        def loss_to_normalized_acc(loss):
            MEAN = 0.908192
            STD = 0.023961
            acc = 1 - loss / 100
            return (acc - MEAN) / STD

        op_map = [OUTPUT, INPUT, *OPS]
        ops = self.get_op_list()
        ops = [INPUT, *ops, OUTPUT]
        ops_onehot = np.array([[i == op_map.index(op) for i in range(len(op_map))] for op in ops], dtype=np.float32)
        val_loss = self.get_val_loss(nasbench, deterministic=deterministic)    
        test_loss = self.get_test_loss(nasbench)
        matrix = np.array(
            [[0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]])

        dic = {
            'num_vertices': 8,
            'adjacency': matrix,
            'operations': ops_onehot,
            'mask': np.array([i < 8 for i in range(8)], dtype=np.float32),
            'val_acc': loss_to_normalized_acc(val_loss),
            'test_acc': loss_to_normalized_acc(test_loss)
        }

        return dic

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

    def get_random_hash(self):
        num = self.get_num()
        hashes = pickle.load(open('nas_bench_201/random_hash.pkl', 'rb'))
        return hashes[num]

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

    def perturb(self, 
                nasbench,
                mutation_rate=1):
        # deterministic version of mutate
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
               mutate_encoding='adj',
               index_hash=None,
               cutoff=30,
               patience=5000):
        p = 0
        if mutate_encoding == 'adj':
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
        
        elif mutate_encoding in ['path', 'trunc_path']:
            path_blueprints = [[3], [0,4], [1,5], [0,2,5]]

            if mutate_encoding == 'trunc_path':
                choice = np.random.choice(range(2))
            else:
                choice = np.random.choice(range(3))

            blueprint = path_blueprints[choice]
            ops = self.get_op_list()
            new_ops = ops.copy()

            for idx in blueprint:
                available = [o for o in OPS if o != ops[idx]]
                new_ops[idx] = np.random.choice(available)

            new_arch = {'string':self.get_string_from_ops(new_ops)}
            return {'string':self.get_string_from_ops(new_ops)}
        else:
            print('{} is an invalid mutate encoding'.format(mutate_encoding))
            raise NotImplementedError()

    def encode_standard(self):
        """ 
        compute the standard encoding
        """
        ops = self.get_op_list()
        encoding = []
        for op in ops:
            encoding.append(OPS.index(op))
        return encoding

    def encode_one_hot(self):
        """
        compute the one-hot encoding
        """
        encoding = self.encode_standard()
        one_hot = []
        for num in encoding:
            for i in range(len(OPS)):
                if i == num:
                    one_hot.append(1)
                else:
                    one_hot.append(0)
        return one_hot

    def get_num_params(self, nasbench):
        # todo: add this method
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

    def encode_freq_paths(self, cutoff=30):
        # natural cutoffs 5, 30, 155 (last)
        num_paths = sum([NUM_OPS ** i for i in range(1, LONGEST_PATH_LENGTH + 1)])
        path_indices = self.get_path_indices()
        encoding = np.zeros(cutoff)
        for index in range(min(num_paths, cutoff)):
            if index in path_indices:
                encoding[index] = 1
        return encoding

    def distance(self, other, dist_type, cutoff=30):
        if dist_type == 'adj':
            distance = adj_distance(self, other)
        elif dist_type == 'path':
            distance = path_distance(self, other)        
        elif dist_type == 'trunc_path':
            distance = path_distance(self, other, cutoff=cutoff)
        elif dist_type == 'nasbot':
            distance = nasbot_distance(self, other)
        else:
            print('{} is an invalid distance'.format(distance))
            raise NotImplementedError()
        return distance


    def get_neighborhood(self, 
                         nasbench, 
                         mutate_encoding,
                         shuffle=True):
        nbhd = []
        ops = self.get_op_list()

        if mutate_encoding == 'adj':
            for i in range(len(ops)):
                available = [op for op in OPS if op != ops[i]]
                for op in available:
                    new_ops = ops.copy()
                    new_ops[i] = op
                    new_arch = {'string':self.get_string_from_ops(new_ops)}
                    nbhd.append(new_arch)

        elif mutate_encoding in ['path', 'trunc_path']:

            if mutate_encoding == 'trunc_path':
                path_blueprints = [[3], [0,4], [1,5]]
            else:
                path_blueprints = [[3], [0,4], [1,5], [0,2,5]]
            ops = self.get_op_list()

            for blueprint in path_blueprints:
                for new_path in itertools.product(OPS, repeat=len(blueprint)):
                    new_ops = ops.copy()

                    for i, op in enumerate(new_path):
                        new_ops[blueprint[i]] = op

                        # check if it's the same
                        same = True
                        for j in range(len(ops)):
                            if ops[j] != new_ops[j]:
                                same = False
                        if not same:
                            new_arch = {'string':self.get_string_from_ops(new_ops)}
                            nbhd.append(new_arch)
        else:
            print('{} is an invalid mutate encoding'.format(mutate_encoding))
            raise NotImplementedError()

        if shuffle:
            random.shuffle(nbhd)                
        return nbhd

