import numpy as np
import pickle
import sys
import os

if 'search_space' not in os.environ or os.environ['search_space'] == 'nasbench':
    from nasbench import api
    from nas_bench.cell import Cell

elif os.environ['search_space'] == 'darts':
    from darts.arch import Arch

elif os.environ['search_space'][:12] == 'nasbench_201':
    from nas_201_api import NASBench201API as API
    from nas_bench_201.cell import Cell

else:
    print('Invalid search space environ in data.py')
    sys.exit()


class Data:

    def __init__(self, 
                 search_space, 
                 dataset='cifar10', 
                 nasbench_folder='./', 
                 loaded_nasbench=None):
        self.search_space = search_space
        self.dataset = dataset

        if loaded_nasbench:
            self.nasbench = loaded_nasbench
        elif search_space == 'nasbench':
                self.nasbench = api.NASBench(nasbench_folder + 'nasbench_only108.tfrecord')
        elif search_space == 'nasbench_201':
            self.nasbench = API(os.path.expanduser('~/nas-bench-201/NAS-Bench-201-v1_0-e61699.pth'))
        elif search_space != 'darts':
            print(search_space, 'is not a valid search space')
            sys.exit()

    def get_type(self):
        return self.search_space

    def query_arch(self, 
                   arch=None, 
                   train=True, 
                   encoding_type='path', 
                   cutoff=-1,
                   deterministic=True, 
                   epochs=0):

        arch_dict = {}
        arch_dict['epochs'] = epochs
        if self.search_space in ['nasbench', 'nasbench_201']:
            if arch is None:
                arch = Cell.random_cell(self.nasbench)

            arch_dict['spec'] = arch

            if encoding_type == 'adj':
                encoding = Cell(**arch).encode_standard()
            elif encoding_type == 'path':
                encoding = Cell(**arch).encode_paths()
            elif encoding_type == 'trunc_path':
                encoding = Cell(**arch).encode_paths()[:cutoff]
            else:
                print('invalid encoding type')

            arch_dict['encoding'] = encoding

            if train:
                arch_dict['val_loss'] = Cell(**arch).get_val_loss(self.nasbench, 
                                                                    deterministic=deterministic,
                                                                    dataset=self.dataset)
                arch_dict['test_loss'] = Cell(**arch).get_test_loss(self.nasbench,
                                                                    dataset=self.dataset)
                arch_dict['num_params'] = Cell(**arch).get_num_params(self.nasbench)
                arch_dict['val_per_param'] = (arch_dict['val_loss'] - 4.8) * (arch_dict['num_params'] ** 0.5) / 100

        else:
            if arch is None:
                arch = Arch.random_arch()

            arch_dict['spec'] = arch

            if encoding_type == 'path':
                encoding = Arch(arch).encode_paths()
            elif encoding_type == 'trunc_path':
                encoding = Arch(arch).encode_paths()[:cutoff]
            else:
                encoding = arch

            arch_dict['encoding'] = encoding

            if train:
                if epochs == 0:
                    epochs = 50
                arch_dict['val_loss'], arch_dict['test_loss'] = Arch(arch).query(epochs=epochs)
        
        return arch_dict           

    def mutate_arch(self, 
                    arch, 
                    mutation_rate=1.0):
        if self.search_space in ['nasbench', 'nasbench_201']:
            return Cell(**arch).mutate(self.nasbench, 
                                       mutation_rate=mutation_rate)
        else:
            return Arch(arch).mutate(int(mutation_rate))

    def get_hash(self, arch):
        # return the path indices of the architecture, used as a hash
        if self.search_space == 'nasbench':
            return Cell(**arch).get_path_indices()
        elif self.search_space == 'darts':
            return Arch(arch).get_path_indices()[0]
        else:
            return Cell(**arch).get_string()

    def generate_random_dataset(self,
                                num=10, 
                                train=True,
                                encoding_type='path', 
                                cutoff=-1,
                                random='standard',
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
            arch_dict = self.query_arch(train=train,
                                        encoding_type=encoding_type,
                                        cutoff=cutoff,
                                        deterministic=deterministic_loss)

            h = self.get_hash(arch_dict['spec'])
            if allow_isomorphisms or h not in dic:
                dic[h] = 1
                data.append(arch_dict)

        return data

    def get_candidates(self, 
                       data, 
                       num=100,
                       acq_opt_type='mutation',
                       encoding_type='path',
                       cutoff=-1,
                       loss='val_loss',
                       patience_factor=5, 
                       deterministic_loss=True,
                       num_arches_to_mutate=1,
                       max_mutation_rate=1,
                       allow_isomorphisms=False):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """

        candidates = []
        # set up hash map
        dic = {}
        for d in data:
            arch = d['spec']
            h = self.get_hash(arch)
            dic[h] = 1

        if acq_opt_type in ['mutation', 'mutation_random']:
            # mutate architectures with the lowest loss
            best_arches = [arch['spec'] for arch in sorted(data, key=lambda i:i[loss])[:num_arches_to_mutate * patience_factor]]

            # stop when candidates is size num
            # use patience_factor instead of a while loop to avoid long or infinite runtime
            for arch in best_arches:
                if len(candidates) >= num:
                    break
                for i in range(num // num_arches_to_mutate // max_mutation_rate):
                    for rate in range(1, max_mutation_rate + 1):
                        mutated = self.mutate_arch(arch, mutation_rate=rate)
                        arch_dict = self.query_arch(mutated,
                                                    train=False,
                                                    encoding_type=encoding_type,
                                                    cutoff=cutoff)
                        h = self.get_hash(mutated)

                        if allow_isomorphisms or h not in dic:
                            dic[h] = 1    
                            candidates.append(arch_dict)

        if acq_opt_type in ['random', 'mutation_random']:
            # add randomly sampled architectures to the set of candidates
            for _ in range(num * patience_factor):
                if len(candidates) >= 2 * num:
                    break

                arch_dict = self.query_arch(train=False, 
                                            encoding_type=encoding_type,
                                            cutoff=cutoff)
                h = self.get_hash(arch_dict['spec'])

                if allow_isomorphisms or h not in dic:
                    dic[h] = 1
                    candidates.append(arch_dict)

        return candidates

    def remove_duplicates(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed

        dic = {}
        for d in data:
            dic[self.get_hash(d['spec'])] = 1
        unduplicated = []
        for candidate in candidates:
            if self.get_hash(candidate['spec']) not in dic:
                dic[self.get_hash(candidate['spec'])] = 1
                unduplicated.append(candidate)
        return unduplicated

    def encode_data(self, dicts):
        """
        method used by metann_runner.py (for Arch)
        input: list of arch dictionary objects
        output: xtrain (encoded architectures), ytrain (val loss)
        """
        data = []

        for dic in dicts:
            arch = dic['spec']
            encoding = Arch(arch).encode_paths()
            data.append((arch, encoding, dic['val_loss_avg'], None))

        return data

    def get_arch_list(self,
                      aux_file_path, 
                      iteridx=0, 
                      num_top_arches=5,
                      max_edits=20, 
                      num_repeats=5,
                      verbose=1):
        # Method used for gp_bayesopt

        if self.search_space == 'darts':
            print('get_arch_list only supported for nasbench and nasbench_201')
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

    @classmethod
    def generate_distance_matrix(cls, arches_1, arches_2, distance):
        # Method used for gp_bayesopt for nasbench
        matrix = np.zeros([len(arches_1), len(arches_2)])
        for i, arch_1 in enumerate(arches_1):
            for j, arch_2 in enumerate(arches_2):
                if distance == 'edit_distance':
                    matrix[i][j] = Cell(**arch_1).edit_distance(Cell(**arch_2))
                elif distance == 'path_distance':
                    matrix[i][j] = Cell(**arch_1).path_distance(Cell(**arch_2))        
                elif distance == 'trunc_path_distance':
                    matrix[i][j] = Cell(**arch_1).path_distance(Cell(**arch_2))        
                elif distance == 'nasbot_distance':
                    matrix[i][j] = Cell(**arch_1).nasbot_distance(Cell(**arch_2))  
                else:
                    print('{} is an invalid distance'.format(distance))
                    sys.exit()
        return matrix
