import sys


def algo_params(param_str):
    """
      Return params list based on param_str.
      These are the parameters used to produce the figures in the paper
      For AlphaX and Reinforcement Learning, we used the corresponding github repos:
      https://github.com/linnanwang/AlphaX-NASBench101
      https://github.com/automl/nas_benchmarks
    """
    params = []

    if param_str == 'main_experiments':
        params.append({'algo_name':'bananas', 'total_queries':150})   
        params.append({'algo_name':'random', 'total_queries':150})
        params.append({'algo_name':'evolution', 'total_queries':150})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':150})        

    elif param_str == 'ablation':
        params.append({'algo_name':'bananas', 'total_queries':150})   
        params.append({'algo_name':'bananas', 'total_queries':150, 'encode_paths':False})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':150, 'distance':'path_distance'})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':150, 'distance':'edit_distance'})
        params.append({'algo_name':'bananas', 'total_queries':150, 'acq_opt_type':'random'})

    elif param_str == 'acq_fn':
        params.append({'algo_name':'bananas', 'total_queries':150, 'explore_type':'ei'})
        params.append({'algo_name':'bananas', 'total_queries':150, 'explore_type':'pi'})
        params.append({'algo_name':'bananas', 'total_queries':150, 'explore_type':'ts'})

    elif param_str == '500_queries':
        params.append({'algo_name':'random', 'total_queries':500})
        params.append({'algo_name':'evolution', 'total_queries':500})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':500})
        params.append({'algo_name':'bananas', 'total_queries':500})

    elif param_str == 'random_validation_error':
        params.append({'algo_name':'random', 'total_queries':150, 'deterministic':False})
        params.append({'algo_name':'evolution', 'total_queries':150, 'deterministic':False})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':150, 'deterministic':False})
        params.append({'algo_name':'bananas', 'total_queries':150, 'deterministic':False})

    elif param_str == 'test':
        params.append({'algo_name':'bananas', 'total_queries':30})   
        params.append({'algo_name':'random', 'total_queries':30})
        params.append({'algo_name':'evolution', 'total_queries':30})
        params.append({'algo_name':'gp_bayesopt', 'total_queries':30})

    else:
        print('invalid algorithm params')
        sys.exit()

    print('\n* Running experiment: ' + param_str)
    return params


def meta_neuralnet_params(param_str):

    if param_str == 'nasbench':
        params = {'search_space':'nasbench', 'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}

    elif param_str == 'darts':
        params = {'search_space':'darts', 'loss':'mape', 'num_layers':10, 'layer_width':20, \
            'epochs':10000, 'batch_size':32, 'lr':.00001, 'regularization':0, 'verbose':0}

    else:
        print('invalid meta neural net params')
        sys.exit()

    return params