import sys
import numpy as np


def algo_params(param_str, queries=150):
    """
      Return params list based on param_str.
      
      TODO: three algorithms, bonas, vaenas, and gcn_predictor are currently
            not compatible with the requirements needed to install nasbench301.
            
            To run these algorithms with nasbench101 or 201, downgrade torch to 1.4.0
    """
    params = []


    if param_str == 'simple_algos':
        params.append({'algo_name':'random', 'total_queries':queries})
        params.append({'algo_name':'evolution', 'total_queries':queries})
        params.append({'algo_name':'local_search', 'total_queries':queries, 'stop_at_minimum':False})
        
    elif param_str == 'all_algos':
        params.append({'algo_name':'random', 'total_queries':queries})
        params.append({'algo_name':'evolution', 'total_queries':queries})
        params.append({'algo_name':'bananas', 'total_queries':queries})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries})
        params.append({'algo_name':'dngo', 'total_queries':queries})
        params.append({'algo_name':'bohamiann', 'total_queries':queries})
        params.append({'algo_name':'local_search', 'total_queries':queries})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'nasbot'})
        #params.append({'algo_name':'gcn_predictor', 'total_queries':queries})
        #params.append({'algo_name':'bonas', 'total_queries':queries})   
        #params.append({'algo_name':'vaenas', 'total_queries':queries})   
        
    elif param_str == 'local_search_variants':
        params.append({'algo_name':'local_search', 'total_queries':queries})
        params.append({'algo_name':'local_search', 'total_queries':queries, 'stop_at_minimum':False})
        params.append({'algo_name':'local_search', 'total_queries':queries, 'query_full_nbhd':True})

    # individual algorithms
    elif param_str == 'random':
        params.append({'algo_name':'random', 'total_queries':queries})
    elif param_str == 'evolution':
        params.append({'algo_name':'evolution', 'total_queries':queries})
    elif param_str == 'bananas':
        params.append({'algo_name':'bananas', 'total_queries':queries})
    elif param_str == 'gp_bo':
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries})
    elif param_str == 'dngo':
        params.append({'algo_name':'dngo', 'total_queries':queries})
    elif param_str == 'bohamiann':
        params.append({'algo_name':'bohamiann', 'total_queries':queries})
    elif param_str == 'local_search':
        params.append({'algo_name':'local_search', 'total_queries':queries})
    elif param_str == 'nasbot':
        params.append({'algo_name':'gp_bayesopt', 'total_queries':queries, 'distance':'nasbot'})
    elif param_str == 'gcn_predictor':
        params.append({'algo_name':'gcn_predictor', 'total_queries':queries})
    elif param_str == 'bonas':
        params.append({'algo_name':'bonas', 'total_queries':queries})   
    elif param_str == 'vaenas':
        params.append({'algo_name':'vaenas', 'total_queries':queries})   
        
    else:
        print('Invalid algorithm params: {}'.format(param_str))
        raise NotImplementedError()
        
    print('\n* Running experiment: ' + param_str)
    return params


def meta_neuralnet_params(param_str):

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
        # TODO: this can be returned without a dictionary (update the algorithms that use metann_params)
        params = {'ensemble_params':ensemble_params}

    else:
        print('Invalid meta neural net params: {}'.format(param_str))
        raise NotImplementedError()

    return params
