import numpy as np
import pickle
from nasbench import api

from nas_bench.cell import Cell


class NasbenchData:

    def __init__(self):
        self.nasbench = api.NASBench('nasbench_only108.tfrecord')

    def get_type(self):
        return 'nasbench'

    def query_arch(self, arch, deterministic=True):
        val_loss = Cell(**arch).get_val_loss(self.nasbench, deterministic)
        test_loss = Cell(**arch).get_test_loss(self.nasbench)
        return (val_loss, test_loss)

    def mutate_arch(self, arch, mutation_rate):
        return Cell(**arch).mutate(self.nasbench, mutation_rate)

    def get_candidates(self, 
                        data, 
                        num=100,
                        allow_isomorphisms=False, 
                        acq_opt_type='mutation',
                        patience_factor=5, 
                        encode_paths=1, 
                        deterministic_loss=True):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
        
        # test for isomorphisms using a hash map of path indices
        candidates = []
        dic = {}
        for d in data:
            arch = d[0]
            path_indices = Cell(**arch).get_path_indices()
            dic[path_indices] = 1            

        if acq_opt_type in ['mutation', 'mutation_random']:
            # mutate architectures with the lowest validation error
            k = int(num * patience_factor / 100)
            best_arches = [arch[0] for arch in sorted(data, key=lambda i:i[2])[:k]]

            # stop when candidates is size num
            # use patience_factor instead of a while loop to avoid long or infinite runtime
            for arch in best_arches:
                if len(candidates) >= num:
                    break
                for i in range(num):
                    mutated = Cell(**arch).mutate(self.nasbench, 1.0)
                    path_indices = Cell(**mutated).get_path_indices()
                    if allow_isomorphisms or path_indices not in dic:
                        dic[path_indices] = 1    

                        if encode_paths:
                            encoding = Cell(**mutated).encode_paths()
                        else:
                            encoding = Cell(**mutated).encode_cell()
                        candidates.append((mutated, encoding))

        if acq_opt_type in ['random', 'mutation_random']:
            # add randomly sampled architectures to the set of candidates
            for _ in range(num * patience_factor):
                if len(candidates) >= 2 * num:
                    break
                arch = Cell.random_cell(self.nasbench)
                path_indices = Cell(**arch).get_path_indices()
                if allow_isomorphisms or path_indices not in dic:
                    dic[path_indices] = 1

                    if encode_paths:
                        encoding = Cell(**arch).encode_paths()
                    else:
                        encoding = Cell(**arch).encode_cell()
                    candidates.append((arch, encoding))

        return candidates


    def generate_random_dataset(self,
                                num, 
                                allow_isomorphisms=False, 
                                patience_factor=5, 
                                encode_paths=1, 
                                deterministic_loss=True):
        """
        create a random dataset
        test for isomorphisms using a hash map of path indices
        tries_left ensures there is no infinite loop caused by removing isomorphisms
        """

        data = []
        dic = {}
        tries_left = num * patience_factor
        while len(data) < num:
            tries_left -= 1
            if tries_left <= 0:
                break
            arch = Cell.random_cell(self.nasbench)
            val_loss = Cell(**arch).get_val_loss(self.nasbench, deterministic_loss)       
            test_loss = Cell(**arch).get_test_loss(self.nasbench)  
            path_indices = Cell(**arch).get_path_indices()
            if allow_isomorphisms or path_indices not in dic:
                dic[path_indices] = 1
                if encode_paths:
                    encoding = Cell(**arch).encode_paths()
                else:
                    encoding = Cell(**arch).encode_cell()
                data.append((arch, encoding, val_loss, test_loss))

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

    # Method used for gp_bayesopt
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
