import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np

from acquisition_functions import acq_fn
from data import Data
from meta_neural_net import MetaNeuralnet


"""
meta neural net runner is used in run_experiments_parallel

 - loads data by opening k*i pickle files from previous iterations
 - trains a meta neural network and predicts accuracy of all candidates
 - outputs k pickle files of the architecture to be trained next
"""

def run_meta_neuralnet(search_space, dicts,
                        k=10,
                        verbose=1, 
                        num_ensemble=5, 
                        epochs=10000,
                        lr=0.00001,
                        loss='scaled',
                        explore_type='its',
                        explore_factor=0.5):

    # data: list of arch dictionary objects
    # trains a meta neural network
    # returns list of k arch dictionary objects - the k best predicted

    results = []
    meta_neuralnet = MetaNeuralnet()
    data = search_space.encode_data(dicts)
    xtrain = np.array([d[1] for d in data])
    ytrain = np.array([d[2] for d in data])

    candidates = search_space.get_candidates(data, 
                                            acq_opt_type='mutation_random',
                                            encode_paths=True, 
                                            allow_isomorphisms=True,
                                            deterministic_loss=None)

    xcandidates = np.array([c[1] for c in candidates])
    candidates_specs = [c[0] for c in candidates]
    predictions = []

    # train an ensemble of neural networks
    train_error = 0
    for _ in range(num_ensemble):
        meta_neuralnet = MetaNeuralnet()
        train_error += meta_neuralnet.fit(xtrain, ytrain,
                                            loss=loss,
                                            epochs=epochs,
                                            lr=lr)
        predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))
    train_error /= num_ensemble
    if verbose:
        print('Meta neural net train error: {}'.format(train_error))

    sorted_indices = acq_fn(predictions, explore_type)

    top_k_candidates = [candidates_specs[i] for i in sorted_indices[:k]]
    candidates_dict = []
    for candidate in top_k_candidates:
        d = {}
        d['spec'] = candidate
        candidates_dict.append(d)

    return candidates_dict


def run(args):

    save_dir = '{}/'.format(args.experiment_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    query = args.query
    k = args.k
    trained_prefix = args.trained_filename
    untrained_prefix = args.untrained_filename
    threshold = args.threshold

    search_space = Data('darts')

    # if it's the first iteration, choose k arches at random to train
    if query == 0:
        print('about to generate {} random'.format(k))
        data = search_space.generate_random_dataset(num=k, train=False)
        arches = [d['spec'] for d in data]

        next_arches = []
        for arch in arches:
            d = {}
            d['spec'] = arch
            next_arches.append(d)

    else:
        # get the data from prior iterations from pickle files
        data = []
        for i in range(query):

            filepath = '{}{}_{}.pkl'.format(save_dir, trained_prefix, i)
            with open(filepath, 'rb') as f:
                arch = pickle.load(f)
            data.append(arch)

        print('Iteration {}'.format(query))
        print('Data from last round')
        print(data)

        # run the meta neural net to output the next arches
        next_arches = run_meta_neuralnet(search_space, data, k=k)

    print('next batch')
    print(next_arches)

    # output the new arches to pickle files
    for i in range(k):
        index = query + i
        filepath = '{}{}_{}.pkl'.format(save_dir, untrained_prefix, index)
        next_arches[i]['index'] = index
        next_arches[i]['filepath'] = filepath
        with open(filepath, 'wb') as f:
            pickle.dump(next_arches[i], f)


def main(args):

    #set up save dir
    save_dir = './'

    #set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for meta neural net')
    parser.add_argument('--experiment_name', type=str, default='darts_test', help='Folder for input/output files')
    parser.add_argument('--params', type=str, default='test', help='Which set of params to use')
    parser.add_argument('--query', type=int, default=0, help='Which query is Neural BayesOpt on')
    parser.add_argument('--trained_filename', type=str, default='trained_spec', help='name of input files')
    parser.add_argument('--untrained_filename', type=str, default='untrained_spec', help='name of output files')
    parser.add_argument('--k', type=int, default=10, help='number of arches to train per iteration')
    parser.add_argument('--threshold', type=int, default=20, help='throw out arches with val loss above threshold')

    args = parser.parse_args()
    main(args)