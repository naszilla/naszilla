import numpy as np
import sys
import os

from darts.arch import Arch

sys.path.append(os.path.expanduser('~/darts/cnn'))
from train_class import Train

class DartsData:

    def get_type(self):
        return 'darts'

    def query_arch(self, arch, epochs=50):
        trainer = Train()
        val_accs, test_accs = trainer.main(arch, epochs=epochs)
        return (np.mean(val_accs), test_accs[-1])

    def mutate_arch(self, arch, mutation_rate):
        return Arch.mutate_arch()

    def generate_random_dataset(self, 
                                num=10, 
                                train=True,
                                encode_paths=True,
                                allow_isomorphisms=False,
                                determinstic_loss=True):
        """
        create a random dataset
        """
        arches = []
        for _ in range(num):
            arch = Arch.sample_arch()
            encoding = Arch.encode_paths(arch)
            if train:
                val_acc, test_acc = self.query_arch(arch)
                arches.append((arch, encoding, val_acc, test_acc))
            else:
                arches.append((arch, encoding))
        return arches

    def get_candidates(self, data,
                        acq_opt_type='mutation_random',
                        encode_paths=True,
                        allow_isomorphisms=False,
                        num_arches_to_mutate=10,
                        max_edits=5,
                        num_copies=5,
                        num=100,
                        deterministic_loss=None):
        """
        Creates a set of candidate architectures with mutated and/or random architectures
        """
        mutation_candidates = []
        random_candidates = []
        best_arches = sorted(data, key=lambda i: i[2])

        if acq_opt_type in ['mutation', 'mutation_random']:
            for i in range(min(num_arches_to_mutate, len(data))):
                arch = best_arches[i][0]
                for edits in range(1, max_edits):
                    for _ in range(num_copies):
                        mutated = Arch.mutate_arch(arch, edits)
                        encoding = Arch.encode_paths(mutated)
                        mutation_candidates.append((mutated, encoding))

        if acq_opt_type in ['random', 'mutation_random']:
            random_candidates = self.generate_random_dataset(num, train=False)
    
        candidates = [*mutation_candidates, *random_candidates]
    
        if not allow_isomorphisms:
            candidates = self.remove_duplicates(candidates, [d['spec'] for d in data])

        return candidates

    def remove_duplicates(self, candidates, data):
        # input: two sets of architectues: candidates and data
        # output: candidates with arches from data removed

        dic = {}
        for spec in data:
            dic[get_path_indices(spec)[0]] = 1
        unduplicated = []
        for candidate in candidates:
            if get_path_indices(candidate)[0] not in dic:
                dic[get_path_indices(candidate)[0]] = 1
                unduplicated.append(candidate)
        return unduplicated


    def encode_data(self, dicts):
        # input: list of arch dictionary objects
        # output: xtrain (in binary path encoding), ytrain (val loss)

        data = []

        for dic in dicts:
            arch = dic['spec']
            encoding = Arch.encode_paths(arch)
            data.append((arch, encoding, dic['val_loss_avg']))

        return data