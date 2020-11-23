import numpy as np
import copy
import itertools
import random
import pickle
import torch

from nasbench import api

from naszilla.nas_bench_101.distances import *
from naszilla.nas_bench_101.sample_random import *
from naszilla.nas_bench_101.encodings import *
from naszilla.nas_bench_101.mutations import *


INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
OPS_INCLUSIVE = [INPUT, OUTPUT, *OPS]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class Cell101:

    def __init__(self, matrix, ops):

        self.matrix = matrix
        self.ops = ops

    def get_matrix(self):
        return self.matrix

    def get_ops(self):
        return self.ops

    def serialize(self):
        return {
            'matrix': self.matrix,
            'ops': self.ops
        }

    def get_utilized(self):
        # return the sets of utilized edges and nodes
        # first, compute all paths
        n = np.shape(self.matrix)[0]
        sub_paths = []
        for j in range(0, n):
            sub_paths.append([[(0, j)]]) if self.matrix[0][j] else sub_paths.append([])
        
        # create paths sequentially
        for i in range(1, n - 1):
            for j in range(1, n):
                if self.matrix[i][j]:
                    for sub_path in sub_paths[i]:
                        sub_paths[j].append([*sub_path, (i, j)])
        paths = sub_paths[-1]

        utilized_edges = []
        for path in paths:
            for edge in path:
                if edge not in utilized_edges:
                    utilized_edges.append(edge)

        utilized_nodes = []
        for i in range(NUM_VERTICES):
            for edge in utilized_edges:
                if i in edge and i not in utilized_nodes:
                    utilized_nodes.append(i)

        return utilized_edges, utilized_nodes

    def num_edges_and_vertices(self):
        # return the true number of edges and vertices
        edges, nodes = self.get_utilized()
        return len(edges), len(nodes)

    def is_valid_vertex(self, vertex):
        edges, nodes = self.get_utilized()
        return (vertex in nodes)

    def is_valid_edge(self, edge):
        edges, nodes = self.get_utilized()
        return (edge in edges)

    @classmethod
    def convert_to_cell(cls, arch):
        matrix, ops = arch['matrix'], arch['ops']

        if len(matrix) < 7:
            # the nasbench spec can have an adjacency matrix of n x n for n<7, 
            # but in the nasbench api, it is always 7x7 (possibly containing blank rows)
            # so this method will add a blank row/column

            new_matrix = np.zeros((7, 7), dtype='int8')
            new_ops = []
            n = matrix.shape[0]
            for i in range(7):
                for j in range(7):
                    if j < n - 1 and i < n:
                        new_matrix[i][j] = matrix[i][j]
                    elif j == n - 1 and i < n:
                        new_matrix[i][-1] = matrix[i][j]

            for i in range(7):
                if i < n - 1:
                    new_ops.append(ops[i])
                elif i < 6:
                    new_ops.append('conv3x3-bn-relu')
                else:
                    new_ops.append('output')
            return {
                'matrix': new_matrix,
                'ops': new_ops
            }

        else:
            return {
                'matrix': matrix,
                'ops': ops
            }

    def encode(self, predictor_encoding, nasbench=None, deterministic=True, cutoff=None):

        if predictor_encoding in ['adj', 'bohamiann']:
            return encode_adj(self.matrix, self.ops)
        elif predictor_encoding == 'cat_adj':
            return encode_cat_adj(self.matrix, self.ops)
        elif predictor_encoding == 'cont_adj':
            return encode_cont_adj(self.matrix, self.ops)
        elif predictor_encoding == 'path':
            return encode_paths(self.get_path_indices())
        elif predictor_encoding == 'trunc_path':
            if not cutoff:
                cutoff = 40
            return encode_paths(self.get_path_indices())[:cutoff]
        elif predictor_encoding == 'cat_path':
            indices = self.get_path_indices()
            return tuple([*indices, *[0]*(20-len(indices))])
        elif predictor_encoding == 'trunc_cat_path':
            if not cutoff:
                cutoff = 40
            indices = [i for i in self.get_path_indices() if i < cutoff]
            return tuple([*indices, *[0]*(20-len(indices))])
        elif predictor_encoding == 'gcn':
            return self.gcn_encoding(nasbench, deterministic=deterministic)
        elif predictor_encoding == 'vae':
            return self.vae_encoding(nasbench, deterministic=deterministic)
        else:
            print('{} is an invalid predictor encoding'.format(predictor_encoding))
            raise NotImplementedError()

    def gcn_encoding(self, nasbench, deterministic):

        def loss_to_normalized_acc(loss):
            MEAN = 0.908192
            STD = 0.023961
            acc = 1 - loss / 100
            normalized = (acc - MEAN) / STD
            return torch.tensor(normalized, dtype=torch.float32)

        op_map = [OUTPUT, INPUT, *OPS]
        ops_onehot = np.array([[i == op_map.index(op) for i in range(len(op_map))] for op in self.ops], dtype=np.float32)
        val_loss = self.get_val_loss(nasbench, deterministic=deterministic)
        test_loss = self.get_test_loss(nasbench)

        dic = {
            'num_vertices': 7,
            'adjacency': self.matrix,
            'operations': ops_onehot,
            'mask': np.array([i < 7 for i in range(7)], dtype=np.float32),
            'val_acc': loss_to_normalized_acc(val_loss),
            'test_acc': loss_to_normalized_acc(test_loss)
        }
        return dic

    def vae_encoding(self, nasbench, deterministic):
        adj_list = [[0 for _ in range(i+1)] for i in range(NUM_VERTICES)]

        # add the node label
        for i in range(NUM_VERTICES):
            adj_list[i][0] = OPS_INCLUSIVE.index(self.ops[i])

        # add the from-edges to each adjacency list
        for i in range(NUM_VERTICES):
            for j in range(i, NUM_VERTICES):
                if self.matrix[i][j]:
                    adj_list[j][i+1] = 1

        acc = 1 - self.get_val_loss(nasbench, deterministic=deterministic) / 100
        return [adj_list, acc]

    def distance(self, other, dist_type, cutoff=None):
        if dist_type == 'adj':
            return adj_distance(self, other)
        elif dist_type == 'cont_adj':
            return cont_adj_distance(self, other)
        elif dist_type == 'path':
            return path_distance(self, other)
        elif dist_type == 'trunc_path':
            if not cutoff:
                cutoff = 40
            return path_distance(self, other, cutoff=cutoff)
        elif dist_type == 'cont_path':
            return cont_path_distance(self, other)
        elif dist_type == 'trunc_cont_path':
            if not cutoff:
                cutoff = 40
            return cont_path_distance(self, other, cutoff=cutoff)
        elif dist_type == 'nasbot':
            return nasbot_distance(self, other)
        else:
            print('{} is an invalid distance'.format(dist_type))
            raise NotImplementedError()

    def mutate(self, 
               nasbench,
               mutation_rate=1.0,
               mutate_encoding='adj',
               cutoff=None,
               comparisons=2500,
               patience=5000,
               prob_wt=False,
               index_hash=None):

        if mutate_encoding in ['adj', 'cont_adj']:
            cont = ('cont' in mutate_encoding)
            return adj_mutate(nasbench,
                              matrix=self.matrix,
                              ops=self.ops,
                              cont=cont)
        elif mutate_encoding == 'trunc_adj':
            return trunc_adj_mutate(nasbench,
                                    self.matrix,
                                    self.ops,
                                    cutoff)
        elif mutate_encoding[-4:] == 'path':
            weighted = ('wtd' in mutate_encoding)
            cont = ('cont' in mutate_encoding)
            if 'trunc' in mutate_encoding and not cutoff:
                cutoff = 40
            else:
                cutoff = None
            path_indices = self.get_path_indices()
            return path_mutate(nasbench,
                               path_indices=path_indices,
                               index_hash=index_hash,
                               cont=cont,
                               weighted=weighted,
                               cutoff=cutoff)

    @classmethod
    def random_cell(cls, 
                    nasbench, 
                    random_encoding, 
                    cutoff=None,
                    max_edges=10, 
                    max_nodes=8,
                    index_hash=None):
        if random_encoding == 'constrained':
            return random_cell_constrained(nasbench, 
                                           max_edges=max_edges, 
                                           max_nodes=max_nodes)
        elif random_encoding == 'uniform':
            return random_cell_uniform(cls, nasbench)
        elif random_encoding == 'adj':
            return random_cell_adj(nasbench)
        elif random_encoding == 'cont_adj':
            return random_cell_cont_adj(nasbench)
        elif random_encoding[-4:] == 'path':
            weighted = ('wtd' in random_encoding)
            cont = ('cont' in random_encoding)
            if 'trunc' in random_encoding and not cutoff:
                cutoff = 40
            elif 'trunc' not in random_encoding:
                cutoff = None

            return random_cell_path(nasbench, 
                                    index_hash=index_hash, 
                                    weighted=weighted,
                                    cont=cont,
                                    cutoff=cutoff)
        else:
            print('{} is an invalid random encoding'.format(random_encoding))
            raise NotImplementedError()

    def get_neighborhood(self,
                         nasbench, 
                         mutate_encoding='adj',
                         cutoff=None,
                         index_hash=None, 
                         shuffle=True):
        if mutate_encoding == 'adj':
            return self.adj_neighborhood(nasbench,
                                         shuffle=shuffle)
        elif mutate_encoding in ['path', 'trunc_path']:
            if 'trunc' in mutate_encoding and not cutoff:
                cutoff = 40
            elif 'trunc' not in mutate_encoding:
                cutoff = None
            return self.path_neighborhood(nasbench,
                                          cutoff=cutoff,
                                          index_hash=index_hash,
                                          shuffle=shuffle)
        else:
            print('{} is an invalid neighborhood encoding'.format(mutate_encoding))
            raise NotImplementedError()

    def get_val_loss(self, nasbench, deterministic=True, patience=50, epochs=None, dataset=None):
        if not deterministic:
            # output one of the three validation accuracies at random
            acc = 0
            if epochs:
                acc = (100*(1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops), epochs=epochs)['validation_accuracy']))
            else:
                acc = (100*(1 - nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']))
            return np.round(acc, 4)
        else:        
            # query the api until we see all three accuracies, then average them
            # a few architectures only have two accuracies, so we use patience to avoid an infinite loop
            accs = []
            while len(accs) < 3 and patience > 0:
                patience -= 1
                if epochs:
                    acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops), epochs=epochs)['validation_accuracy']
                else:
                    acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['validation_accuracy']
                if acc not in accs:
                    accs.append(acc)
            return round(100*(1-np.mean(accs)), 4)  

    def get_test_loss(self, nasbench, patience=50, epochs=None, dataset=None):
        """
        query the api until we see all three accuracies, then average them
        a few architectures only have two accuracies, so we use patience to avoid an infinite loop
        """
        accs = []
        while len(accs) < 3 and patience > 0:
            patience -= 1
            acc = nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['test_accuracy']
            if acc not in accs:
                accs.append(acc)
        return round(100*(1-np.mean(accs)), 4)

    def get_num_params(self, nasbench):
        return nasbench.query(api.ModelSpec(matrix=self.matrix, ops=self.ops))['trainable_parameters']

    def perturb(self, nasbench, edits=1):
        """ 
        create new perturbed cell 
        inspird by https://github.com/google-research/nasbench
        """
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


    def get_paths(self):
        """ 
        return all paths from input to output
        """
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
        compute the index of each path
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

        path_indices.sort()
        return tuple(path_indices)

    def adj_neighborhood(self, nasbench, shuffle=True):
        nbhd = []
        # add op neighbors
        for vertex in range(1, OP_SPOTS + 1):
            if self.is_valid_vertex(vertex):
                available = [op for op in OPS if op != self.ops[vertex]]
                for op in available:
                    new_matrix = copy.deepcopy(self.matrix)
                    new_ops = copy.deepcopy(self.ops)
                    new_ops[vertex] = op
                    new_arch = {'matrix':new_matrix, 'ops':new_ops}
                    nbhd.append(new_arch)

        # add edge neighbors
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src+1, NUM_VERTICES):
                new_matrix = copy.deepcopy(self.matrix)
                new_ops = copy.deepcopy(self.ops)
                new_matrix[src][dst] = 1 - new_matrix[src][dst]
                new_arch = {'matrix':new_matrix, 'ops':new_ops}
            
                if self.matrix[src][dst] and self.is_valid_edge((src, dst)):
                    spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
                    if nasbench.is_valid(spec):                            
                        nbhd.append(new_arch)  

                if not self.matrix[src][dst] and Cell101(**new_arch).is_valid_edge((src, dst)):
                    spec = api.ModelSpec(matrix=new_matrix, ops=new_ops)
                    if nasbench.is_valid(spec):                            
                        nbhd.append(new_arch)            

        if shuffle:
            random.shuffle(nbhd)
        return nbhd

    def path_neighborhood(self, 
                          nasbench,
                          cutoff,
                          index_hash,
                          shuffle):
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
            
        nbhd = []
        path_indices = self.get_path_indices()
        total_paths = sum([len(OPS) ** i for i in range(OP_SPOTS + 1)])

        if cutoff:
            cutoff_value = cutoff
        else:
            cutoff_value = total_paths

        new_sets = []
        path_indices_cutoff = [path for path in path_indices if path < cutoff_value]

        # remove paths
        for path in path_indices_cutoff:
            new_path_indices = [p for p in path_indices if p != path]
            new_sets.append(new_path_indices)

        # add paths
        other_paths = [path for path in range(cutoff_value) if path not in path_indices]
        for path in other_paths:
            new_path_indices = [*path_indices, path]
            new_sets.append(new_path_indices)

        for new_path_indices in new_sets:
            new_tuple = tuple(new_path_indices)
            if new_tuple in index_hash:

                spec = index_hash[new_tuple]
                matrix = spec['matrix']
                ops = spec['ops']
                model_spec = api.ModelSpec(matrix=matrix, ops=ops)
                if nasbench.is_valid(model_spec):                            
                    nbhd.append(spec)

        if shuffle:
            random.shuffle(nbhd)
        return nbhd

