import numpy as np
import pickle

from nasbench import api
from nas_bench.cell import Cell
from darts.arch import Arch


class Data:

    def __init__(self, search_space):
        self.search_space = search_space
        if search_space == 'nasbench':
            self.nasbench = api.NASBench('nasbench_only108.tfrecord')

    def get_type(self):
        return self.search_space

    def query_arch(self, 
                    arch=None, 
                    train=True, 
                    encode_paths=True, 
                    deterministic=True, 
                    epochs=50):

        if self.search_space == 'nasbench':
            if arch is None:
                arch = Cell.random_cell(self.nasbench)
            if encode_paths:
                encoding = Cell(**arch).encode_paths()
            else:
                encoding = Cell(**arch).encode_cell()

            if train:
                val_loss = Cell(**arch).get_val_loss(self.nasbench, deterministic)
                test_loss = Cell(**arch).get_test_loss(self.nasbench)
                return (arch, encoding, val_loss, test_loss)
            else:
                return (arch, encoding)
        else:
            if arch is None:
                arch = Arch.random_arch()
            if encode_paths:
                encoding = Arch(arch).encode_paths()
            else:
                encoding = arch
                        
            if train:
                val_loss, test_loss = Arch(arch).query(epochs=epochs)
                return (arch, encoding, val_loss, test_loss)
            else:
                return (arch, encoding)

    def mutate_arch(self, arch, mutation_rate=1.0):
        if self.search_space == 'nasbench':
            return Cell(**arch).mutate(self.nasbench, mutation_rate)
        else:
            return Arch(arch).mutate(int(mutation_rate))

    def get_path_indices(self, arch):
        if self.search_space == 'nasbench':
            return Cell(**arch).get_path_indices()
        else:
            return Arch(arch).get_path_indices()[0]

    def generate_random_dataset(self,
                                num=10, 
                                train=True,
                                encode_paths=True, 
                                allow_isomorphisms=False, 
                                deterministic_loss=True,
                                patience_factor=5):
        """
        create a dataset of randomly sampled architectues
        test for isomorphisms using a hash map of path indices
        use patience_factor to avoid infinite loops
        """
        data = []
        dic = {}
        tries_left = num * patience_factor
        while len(data) < num:
            tries_left -= 1
            if tries_left <= 0:
                break
            archtuple = self.query_arch(train=train,
                                        encode_paths=encode_paths,
                                        deterministic=deterministic_loss)
            path_indices = self.get_path_indices(archtuple[0])

            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                data.append(archtuple)

        return data

    def get_candidates(self, data, 
                        num=100,
                        acq_opt_type='mutation',
                        encode_paths=True, 
                        allow_isomorphisms=False, 
                        patience_factor=5, 
                        deterministic_loss=True,
                        num_best_arches=10):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
        
        # test for isomorphisms using a hash map of path indices
        candidates = []
        dic = {}
        for d in data:
            arch = d[0]
            path_indices = self.get_path_indices(arch)
            dic[path_indices] = 1            

        if acq_opt_type in ['mutation', 'mutation_random']:
            # mutate architectures with the lowest validation error
            best_arches = [arch[0] for arch in sorted(data, key=lambda i:i[2])[:num_best_arches * patience_factor]]

            # stop when candidates is size num
            # use patience_factor instead of a while loop to avoid long or infinite runtime
            for arch in best_arches:
                if len(candidates) >= num:
                    break
                for i in range(num):
                    mutated = self.mutate_arch(arch)
                    archtuple = self.query_arch(mutated, 
                                                train=False,
                                                encode_paths=encode_paths)
                    path_indices = self.get_path_indices(mutated)

                    if allow_isomorphisms or path_indices not in dic:
                        dic[path_indices] = 1    
                        candidates.append(archtuple)

        if acq_opt_type in ['random', 'mutation_random']:
            # add randomly sampled architectures to the set of candidates
            for _ in range(num * patience_factor):
                if len(candidates) >= 2 * num:
                    break

                archtuple = self.query_arch(train=False, encode_paths=encode_paths)
                path_indices = self.get_path_indices(archtuple[0])

                if allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1
                    candidates.append(archtuple)

        return candidates


    def remove_duplicates(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed

        dic = {}
        for d in data:
            dic[self.get_path_indices(d[0])] = 1
        unduplicated = []
        for candidate in candidates:
            if self.get_path_indices(candidate[0]) not in dic:
                dic[self.get_path_indices(candidate[0])] = 1
                unduplicated.append(candidate)
        return unduplicated


    def encode_data(self, dicts):
        # input: list of arch dictionary objects
        # output: xtrain (in binary path encoding), ytrain (val loss)

        data = []

        for dic in dicts:
            arch = dic['spec']
            encoding = Arch(arch).encode_paths()
            data.append((arch, encoding, dic['val_loss_avg'], None))

        return data

    # Method used for gp_bayesopt
    def get_arch_list(self,
                        aux_file_path, 
                        distance=None, 
                        iteridx=0, 
                        num_top_arches=5,
                        max_edits=20, 
                        num_repeats=5,
                        verbose=1):

        if self.search_space != 'nasbench':
            print('get_arch_list only supported for nasbench search space')
            sys.exit()

        # load the list of architectures chosen by bayesopt so far
        base_arch_list = pickle.load(open(aux_file_path, 'rb'))
        top_arches = [archtuple[0] for archtuple in base_arch_list[:num_top_arches]]
        if verbose:
            top_5_loss = [archtuple[1][0] for archtuple in base_arch_list[:min(5, len(base_arch_list))]]
            print('top 5 val losses {}'.format(top_5_loss))

        # perturb the best k architectures    
        dic = {}
        for archtuple in base_arch_list:
            path_indices = Cell(**archtuple[0]).get_path_indices()
            dic[path_indices] = 1

        new_arch_list = []
        for arch in top_arches:
            for edits in range(1, max_edits):
                for _ in range(num_repeats):
                    perturbation = Cell(**arch).perturb(self.nasbench, edits)
                    path_indices = Cell(**perturbation).get_path_indices()
                    if path_indices not in dic:
                        dic[path_indices] = 1
                        new_arch_list.append(perturbation)

        # make sure new_arch_list is not empty
        while len(new_arch_list) == 0:
            for _ in range(100):
                arch = Cell.random_cell(self.nasbench)
                path_indices = Cell(**arch).get_path_indices()
                if path_indices not in dic:
                    dic[path_indices] = 1
                    new_arch_list.append(arch)

        return new_arch_list

    # Method used for gp_bayesopt for nasbench
    @classmethod
    def generate_distance_matrix(cls, arches_1, arches_2, distance):
        matrix = np.zeros([len(arches_1), len(arches_2)])
        for i, arch_1 in enumerate(arches_1):
            for j, arch_2 in enumerate(arches_2):
                if distance == 'edit_distance':
                    matrix[i][j] = Cell(**arch_1).edit_distance(Cell(**arch_2))
                elif distance == 'path_distance':
                    matrix[i][j] = Cell(**arch_1).path_distance(Cell(**arch_2))        
                else:
                    print('{} is an invalid distance'.format(distance))
                    sys.exit()
        return matrix
