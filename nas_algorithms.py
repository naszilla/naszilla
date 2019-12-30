import itertools
import os
import pickle
import sys
import copy
import numpy as np
import random
import tensorflow as tf
from argparse import Namespace

from data import Data
from acquisition_functions import acq_fn
from meta_neural_net import MetaNeuralnet
from bo.bo.probo import ProBO

def run_nas_algorithm(algo_params, metann_params):

    # set up search space
    mp = copy.deepcopy(metann_params)
    ss = mp.pop('search_space')
    search_space = Data(ss)

    # run nas algorithm
    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')

    if algo_name == 'random':
        data = random_search(search_space, **ps)
    elif algo_name == 'evolution':
        data = evolution_search(search_space, **ps)
    elif algo_name == 'bananas':
        data = bananas(search_space, mp, **ps)
    elif algo_name == 'gp_bayesopt':
        data = gp_bayesopt_search(search_space, **ps)
    else:
        print('invalid algorithm name')
        sys.exit()

    k = 10
    if 'k' in ps:
        k = ps['k']

    return compute_best_test_losses(data, k, ps['total_queries'])


def compute_best_test_losses(data, k, total_queries):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error 
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i:i[2])[0]
        test_error = best_arch[3]
        results.append((query, test_error))

    return results


def random_search(search_space,
                    total_queries=100, 
                    k=10,
                    allow_isomorphisms=False, 
                    deterministic=True,
                    verbose=1):
    """ 
    random search
    """
    data = search_space.generate_random_dataset(num=total_queries, 
                                                allow_isomorphisms=allow_isomorphisms, 
                                                deterministic_loss=deterministic)
    
    if verbose:
        top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
        print('Query {}, top 5 val losses {}'.format(total_queries, top_5_loss))    
    return data


def evolution_search(search_space,
                        num_init=10,
                        k=10,
                        population_size=30,
                        total_queries=100,
                        tournament_size=10,
                        mutation_rate=1.0, 
                        allow_isomorphisms=False,
                        deterministic=True,
                        verbose=1):
    """
    regularized evolution
    """
    data = search_space.generate_random_dataset(num=num_init, 
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    val_losses = [d[2] for d in data]
    query = num_init

    if num_init <= population_size:
        population = [i for i in range(num_init)]
    else:
        population = np.argsort(val_losses)[:population_size]

    while query <= total_queries:

        # evolve the population by mutating the best architecture
        # from a random subset of the population
        sample = random.sample(population, tournament_size)
        best_index = sorted([(i, val_losses[i]) for i in sample], key=lambda i:i[1])[0][0]
        mutated = search_space.mutate_arch(data[best_index][0], mutation_rate)
        archtuple = search_space.query_arch(mutated, deterministic=deterministic)
        
        data.append(archtuple)
        val_losses.append(archtuple[2])
        population.append(len(data) - 1)

        # kill the worst from the population
        if len(population) >= population_size:
            worst_index = sorted([(i, val_losses[i]) for i in population], key=lambda i:i[1])[-1][0]
            population.remove(worst_index)

        if verbose and (query % k == 0):
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            print('Query {}, top 5 val losses {}'.format(query, top_5_loss))

        query += 1

    return data


def bananas(search_space, metann_params,
            num_init=10, 
            k=10, 
            total_queries=150, 
            num_ensemble=5, 
            acq_opt_type='mutation',
            explore_type='its',
            encode_paths=True,
            allow_isomorphisms=False,
            deterministic=True,
            verbose=1):
    """
    Bayesian optimization with a neural network model
    """

    data = search_space.generate_random_dataset(num=num_init, 
                                                encode_paths=encode_paths, 
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)
    query = num_init + k

    while query <= total_queries:

        xtrain = np.array([d[1] for d in data])
        ytrain = np.array([d[2] for d in data])

        candidates = search_space.get_candidates(data, 
                                                acq_opt_type=acq_opt_type,
                                                encode_paths=encode_paths, 
                                                allow_isomorphisms=allow_isomorphisms,
                                                deterministic_loss=deterministic)

        xcandidates = np.array([c[1] for c in candidates])
        predictions = []

        # train an ensemble of neural networks
        train_error = 0
        for _ in range(num_ensemble):
            meta_neuralnet = MetaNeuralnet()
            train_error += meta_neuralnet.fit(xtrain, ytrain, **metann_params)

            # predict the validation loss of the candidate architectures
            predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))

            # clear the tensorflow graph
            tf.reset_default_graph()

        train_error /= num_ensemble
        if verbose:
            print('Query {}, Meta neural net train error: {}'.format(query, train_error))

        # compute the acquisition function for all the candidate architectures
        sorted_indices = acq_fn(predictions, explore_type)

        # add the k arches with the minimum acquisition function values
        for i in sorted_indices[:k]:
            archtuple = search_space.query_arch(candidates[i][0],
                                                encode_paths=encode_paths,
                                                deterministic=deterministic)
            data.append(archtuple)

        if verbose:
            top_5_loss = sorted([d[2] for d in data])[:min(5, len(data))]
            print('Query {}, top 5 val losses {}'.format(query, top_5_loss))

        query += k

    return data

def gp_bayesopt_search(search_space,
                        num_init=10,
                        k=10,
                        total_queries=100,
                        distance='edit_distance',
                        deterministic=True,
                        tmpdir='./',
                        max_iter=200,
                        mode='single_process',
                        nppred=1000):
    """
    Bayesian optimization with a GP prior
    """
    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        return search_space.query_arch(arch, deterministic=deterministic)[2]

    # set up the path for auxiliary pickle files
    aux_file_path = os.path.join(tmpdir, 'aux.pkl')

    # set all the parameters for the various BayesOpt classes
    fhp = Namespace(fhstr='object', namestr='train')
    domp = Namespace(dom_str='list', set_domain_list_auto=True,
                     aux_file_path=aux_file_path,
                     distance=distance)
    modelp = Namespace(kernp=Namespace(ls=3., alpha=1.5, sigma=1e-5),
                       infp=Namespace(niter=num_iterations, nwarmup=500),
                       distance=distance, search_space=search_space.get_type())
    amp = Namespace(am_str='mygpdistmat_ucb', nppred=nppred, modelp=modelp)
    optp = Namespace(opt_str='rand', max_iter=max_iter)
    makerp = Namespace(domp=domp, amp=amp, optp=optp)
    probop = Namespace(niter=num_iterations, fhp=fhp,
                       makerp=makerp, tmpdir=tmpdir, mode=mode)
    data = Namespace()

    # Set up initial data
    init_data = search_space.generate_random_dataset(num=num_init, 
                                                    deterministic_loss=deterministic)
    data.X = [d[0] for d in init_data]
    data.y = np.array([[d[2]] for d in init_data])

    # initialize aux file
    pairs = [(data.X[i], data.y[i]) for i in range(len(data.y))]
    pairs.sort(key=lambda x: x[1])
    with open(aux_file_path, 'wb') as f:
        pickle.dump(pairs, f)

    # run Bayesian Optimization
    bo = ProBO(fn, search_space, aux_file_path, data, probop, True)
    bo.run_bo()

    # get the validation and test loss for all architectures chosen by BayesOpt
    results = []
    for arch in data.X:
        archtuple = search_space.query_arch(arch)
        results.append(archtuple)

    return results


