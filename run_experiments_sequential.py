import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np
import copy

from params import *


def run_experiments(args, save_dir):

    os.environ['search_space'] = args.search_space

    from nas_algorithms import run_nas_algorithm
    from data import Data

    trials = args.trials
    out_file = args.output_filename
    save_specs = args.save_specs
    metann_params = meta_neuralnet_params(args.search_space)
    algorithm_params = algo_params(args.algo_params)
    num_algos = len(algorithm_params)
    logging.info(algorithm_params)

    # set up search space
    mp = copy.deepcopy(metann_params)
    ss = mp.pop('search_space')
    dataset = mp.pop('dataset')
    search_space = Data(ss, dataset=dataset)

    for i in range(trials):
        results = []
        walltimes = []
        run_data = []

        for j in range(num_algos):
            # run NAS algorithm
            print('\n* Running algorithm: {}'.format(algorithm_params[j]))
            starttime = time.time()
            algo_result, run_datum = run_nas_algorithm(algorithm_params[j], search_space, mp)
            algo_result = np.round(algo_result, 5)

            # remove unnecessary dict entries that take up space
            for d in run_datum:
                if not save_specs:
                    d.pop('spec')
                for key in ['encoding', 'adjacency', 'path', 'dist_to_min']:
                    if key in d:
                        d.pop(key)

            # add walltime, results, run_data
            walltimes.append(time.time()-starttime)
            results.append(algo_result)
            run_data.append(run_datum)

        # print and pickle results
        filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
        print('\n* Trial summary: (params, results, walltimes)')
        print(algorithm_params)
        print(metann_params)
        print(results)
        print(walltimes)
        print('\n* Saving to file {}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump([algorithm_params, metann_params, results, walltimes, run_data], f)
            f.close()

def main(args):

    # make save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    algo_params = args.algo_params
    save_path = save_dir + '/' + algo_params + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_experiments(args, save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for BANANAS experiments')
    parser.add_argument('--trials', type=int, default=500, help='Number of trials')
    parser.add_argument('--search_space', type=str, default='nasbench', \
        help='nasbench or darts')
    parser.add_argument('--algo_params', type=str, default='main_experiments', help='which parameters to use')
    parser.add_argument('--output_filename', type=str, default='round', help='name of output files')
    parser.add_argument('--save_dir', type=str, default='results_output', help='name of save directory')
    parser.add_argument('--save_specs', type=bool, default=False, help='save the architecture specs')    

    args = parser.parse_args()
    main(args)
