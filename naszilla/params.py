import sys
import numpy as np


def algo_params(param_str, queries=150, noise_factor=1):
    """
      Return params list based on param_str.
    """
    params = []


    if param_str == 'fast_algos':
        params.append({'algo_name':'random', 'total_queries':queries, 'noise_factor':noise_factor})
        params.append({'algo_name':'evolution', 'total_queries':queries, 'noise_factor':noise_factor})
        params.append({'algo_name':'local_search', 'total_queries':queries, 'noise_factor':noise_factor, 'stop_at_minimum':False})

    if param_str == 'all_algos':
        params.append({'algo_name':'bananas', 'total_queries':queries})
        params.append({'algo_name':'random', 'total_queries':queries})
        params.append({'algo_name':'evolution', 'total_queries':queries})
        params.append({'algo_name':'local_search', 'total_queries':queries})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'nasbot'})
        params.append({'algo_name':'dngo', 'total_queries':queries})
        params.append({'algo_name':'bohamiann', 'total_queries':queries})
        params.append({'algo_name':'bonas', 'total_queries':queries})   
        params.append({'algo_name':'gcn_predictor', 'total_queries':queries})   
        params.append({'algo_name':'vaenas', 'total_queries':queries})   
        
    elif param_str == 'local_search_variants':
        params.append({'algo_name':'local_search', 'total_queries':queries})
        params.append({'algo_name':'local_search', 'total_queries':queries, 'stop_at_minimum':False})
        params.append({'algo_name':'local_search', 'total_queries':queries, 'query_full_nbhd':True})
        params.append({'algo_name':'local_search', 'total_queries':queries, 'query_full_nbhd':True, 'stop_at_minimum':False})

    else:
        print('invalid algorithm params: {}'.format(param_str))
        sys.exit()

    print('\n* Running experiment: ' + param_str)
    return params


def meta_neuralnet_params(param_str):

    # TODO: don't return a dictionary (and update all algorithms that use metann_params)
    if param_str == 'standard':
        metanet_params = {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
        params = {'ensemble_params':[metanet_params for _ in range(5)]}

    elif param_str == 'diverse':
        metanet_params = {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
        ensemble_params = [
            {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':5, 'layer_width':5, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':5, 'layer_width':30, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':30, 'layer_width':5, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':30, 'layer_width':30, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
        ]
        params = {'ensemble_params':ensemble_params}

    else:
        print('invalid meta neural net params: {}'.format(param_str))
        sys.exit()

    return params
