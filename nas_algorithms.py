import itertools
import os
import pickle
import sys
import copy
import numpy as np
import tensorflow as tf
from argparse import Namespace

from data import Data


def run_nas_algorithm(algo_params, search_space, mp):

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
    elif algo_name == 'dngo':
        data = dngo_search(search_space, **ps)
    else:
        print('invalid algorithm name')
        sys.exit()

    k = 10
    if 'k' in ps:
        k = ps['k']
    total_queries = 150
    if 'total_queries' in ps:
        total_queries = ps['total_queries']
    loss = 'val_loss'
    if 'loss' in ps:
        loss = ps['loss']

    return compute_best_test_losses(data, k, total_queries, loss), data


def compute_best_test_losses(data, k, total_queries, loss):
    """
    Given full data from a completed nas algorithm,
    output the test error of the arch with the best val error 
    after every multiple of k
    """
    results = []
    for query in range(k, total_queries + k, k):
        best_arch = sorted(data[:query], key=lambda i:i[loss])[0]
        test_error = best_arch['test_loss']
        results.append((query, test_error))

    return results


def random_search(search_space,
                  total_queries=150, 
                  loss='val_loss',
                  deterministic=True,
                  verbose=1):
    """ 
    random search
    """
    data = search_space.generate_random_dataset(num=total_queries, 
                                                encoding_type='adj',
                                                deterministic_loss=deterministic)
    
    if verbose:
        top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
        print('random, query {}, top 5 losses {}'.format(total_queries, top_5_loss))    
    return data


def evolution_search(search_space,
                     total_queries=150,
                     num_init=10,
                     k=10,
                     loss='val_loss',
                     population_size=30,                       
                     tournament_size=10,
                     mutation_rate=1.0, 
                     deterministic=True,
                     regularize=True,
                     verbose=1):
    """
    regularized evolution
    """
    data = search_space.generate_random_dataset(num=num_init, 
                                                deterministic_loss=deterministic)

    losses = [d[loss] for d in data]
    query = num_init
    population = [i for i in range(min(num_init, population_size))]

    while query <= total_queries:

        # evolve the population by mutating the best architecture
        # from a random subset of the population
        sample = np.random.choice(population, tournament_size)
        best_index = sorted([(i, losses[i]) for i in sample], key=lambda i:i[1])[0][0]
        mutated = search_space.mutate_arch(data[best_index]['spec'],
                                           mutation_rate=mutation_rate)
        arch_dict = search_space.query_arch(mutated, deterministic=deterministic)
        data.append(arch_dict)        
        losses.append(arch_dict[loss])
        population.append(len(data) - 1)

        # kill the oldest (or worst) from the population
        if len(population) >= population_size:
            if regularize:
                oldest_index = sorted([i for i in population])[0]
                population.remove(oldest_index)
            else:
                worst_index = sorted([(i, losses[i]) for i in population], key=lambda i:i[1])[-1][0]
                population.remove(worst_index)

        if verbose and (query % k == 0):
            top_5_loss = sorted([d[loss] for d in data])[:min(5, len(data))]
            print('evolution, query {}, top 5 losses {}'.format(query, top_5_loss))

        query += 1

    return data


def bananas(search_space, 
            metann_params,
            num_init=10, 
            k=10, 
            loss='val_loss',
            total_queries=150, 
            num_ensemble=5, 
            acq_opt_type='mutation',
            num_arches_to_mutate=1,
            explore_type='its',
            encoding_type='trunc_path',
            cutoff=40,
            deterministic=True,
            verbose=1):
    """
    Bayesian optimization with a neural network model
    """
    from acquisition_functions import acq_fn
    from meta_neural_net import MetaNeuralnet

    data = search_space.generate_random_dataset(num=num_init, 
                                                encoding_type=encoding_type, 
                                                cutoff=cutoff,
                                                deterministic_loss=deterministic)

    query = num_init + k

    while query <= total_queries:

        xtrain = np.array([d['encoding'] for d in data])
        ytrain = np.array([d[loss] for d in data])

        if (query == num_init + k) and verbose:
            print('bananas xtrain shape', xtrain.shape)
            print('bananas ytrain shape', ytrain.shape)

        # get a set of candidate architectures
        candidates = search_space.get_candidates(data, 
                                                 acq_opt_type=acq_opt_type,
                                                 encoding_type=encoding_type, 
                                                 cutoff=cutoff,
                                                 num_arches_to_mutate=num_arches_to_mutate,
                                                 loss=loss,
                                                 deterministic_loss=deterministic)

        xcandidates = np.array([c['encoding'] for c in candidates])
        candidate_predictions = []

        # train an ensemble of neural networks
        train_error = 0
        for _ in range(num_ensemble):
            meta_neuralnet = MetaNeuralnet()
            train_error += meta_neuralnet.fit(xtrain, ytrain, **metann_params)

            # predict the validation loss of the candidate architectures
            candidate_predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))

            # clear the tensorflow graph
            tf.reset_default_graph()

        tf.keras.backend.clear_session()

        train_error /= num_ensemble
        if verbose:
            print('query {}, Meta neural net train error: {}'.format(query, train_error))

        # compute the acquisition function for all the candidate architectures
        candidate_indices = acq_fn(candidate_predictions, explore_type)

        # add the k arches with the minimum acquisition function values
        for i in candidate_indices[:k]:

            arch_dict = search_space.query_arch(candidates[i]['spec'],
                                                encoding_type=encoding_type,
                                                cutoff=cutoff,
                                                deterministic=deterministic)
            data.append(arch_dict)

        if verbose:
            top_5_loss = sorted([(d[loss], d['epochs']) for d in data], key=lambda d: d[0])[:min(5, len(data))]
            print('bananas, query {}, top 5 losses (loss, test, epoch): {}'.format(query, top_5_loss))
            recent_10_loss = [(d[loss], d['epochs']) for d in data[-10:]]
            print('bananas, query {}, most recent 10 (loss, test, epoch): {}'.format(query, recent_10_loss))

        query += k

    return data


def gp_bayesopt_search(search_space,
                        num_init=10,
                        k=10,
                        total_queries=150,
                        distance='edit_distance',
                        deterministic=True,
                        tmpdir='./temp',
                        max_iter=200,
                        mode='single_process',
                        nppred=1000):
    """
    Bayesian optimization with a GP prior
    """
    from bo.bo.probo import ProBO

    # set up the path for auxiliary pickle files
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    aux_file_path = os.path.join(tmpdir, 'aux.pkl')

    num_iterations = total_queries - num_init

    # black-box function that bayesopt will optimize
    def fn(arch):
        return search_space.query_arch(arch, deterministic=deterministic)['val_loss']

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
    data.X = [d['spec'] for d in init_data]
    data.y = np.array([[d['val_loss']] for d in init_data])

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


def dngo_search(search_space,
                num_init=10,
                k=10,
                loss='val_loss',
                total_queries=150,
                encoding_type='path',
                cutoff=40,
                acq_opt_type='mutation',
                explore_type='ucb',
                deterministic=True,
                verbose=True):

    import torch
    from pybnn import DNGO
    from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
    from acquisition_functions import acq_fn

    def fn(arch):
        return search_space.query_arch(arch, deterministic=deterministic)[loss]

    # set up initial data
    data = search_space.generate_random_dataset(num=num_init, 
                                                encoding_type=encoding_type,
                                                cutoff=cutoff,
                                                deterministic_loss=deterministic)

    query = num_init + k

    while query <= total_queries:

        # set up data
        x = np.array([d['encoding'] for d in data])
        y = np.array([d[loss] for d in data])

        # get a set of candidate architectures
        candidates = search_space.get_candidates(data, 
                                                 acq_opt_type=acq_opt_type,
                                                 encoding_type=encoding_type, 
                                                 cutoff=cutoff,
                                                 deterministic_loss=deterministic)

        xcandidates = np.array([d['encoding'] for d in candidates])

        # train the model
        model = DNGO(do_mcmc=False)
        model.train(x, y, do_optimize=True)

        predictions = model.predict(xcandidates)
        candidate_indices = acq_fn(np.array(predictions), explore_type)

        # add the k arches with the minimum acquisition function values
        for i in candidate_indices[:k]:
            arch_dict = search_space.query_arch(candidates[i]['spec'],
                                                encoding_type=encoding_type,
                                                cutoff=cutoff,
                                                deterministic=deterministic)
            data.append(arch_dict)

        if verbose:
            top_5_loss = sorted([(d[loss], d['epochs']) for d in data], key=lambda d: d[0])[:min(5, len(data))]
            print('dngo, query {}, top 5 val losses (val, test, epoch): {}'.format(query, top_5_loss))
            recent_10_loss = [(d[loss], d['epochs']) for d in data[-10:]]
            print('dngo, query {}, most recent 10 (val, test, epoch): {}'.format(query, recent_10_loss))

        query += k

    return data
