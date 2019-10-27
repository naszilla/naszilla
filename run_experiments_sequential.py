import argparse
import time
import logging
import sys
import os
import pickle
import numpy as np

from nas_algorithms import run_nas_algorithm
from params import *


def run_experiments(args, save_dir):

    trials = args.trials
    out_file = args.output_filename
    metann_params = meta_neuralnet_params(args.search_space)
    algorithm_params = algo_params(args.algo_params)
    num_algos = len(algorithm_params)
    logging.info(algorithm_params)

    for i in range(trials):
        results = []
        walltimes = []

        for j in range(num_algos):
            # run NAS algorithm
            print('\n* Running algorithm: {}'.format(algorithm_params[j]))
            starttime = time.time()
            algo_result = run_nas_algorithm(algorithm_params[j], metann_params)
            algo_result = np.round(algo_result, 5)

            # add walltime and results
            walltimes.append(time.time()-starttime)
            results.append(algo_result)

        # print and pickle results
        filename = os.path.join(save_dir, '{}_{}.pkl'.format(out_file, i))
        print('\n* Trial summary: (params, results, walltimes)')
        print(algorithm_params)
        print(metann_params)
        print(results)
        print(walltimes)
        print('\n* Saving to file {}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump([algorithm_params, metann_params, results, walltimes], f)
            f.close()

def main(args):

    # make save directory
    save_dir = args.save_dir
    if not save_dir:
        save_dir = args.algo_params + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run_experiments(args, save_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for BANANAS experiments')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--search_space', type=str, default='nasbench', \
        help='nasbench or darts')
    parser.add_argument('--algo_params', type=str, default='main_experiments', help='which parameters to use')
    parser.add_argument('--output_filename', type=str, default='round', help='name of output files')
    parser.add_argument('--save_dir', type=str, default=None, help='name of save directory')

    args = parser.parse_args()
    main(args)
