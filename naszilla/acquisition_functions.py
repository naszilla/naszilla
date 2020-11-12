import numpy as np
from scipy.stats import norm
import sys

# Different acquisition functions
def acq_fn(predictions, ytrain=None, stds=None, explore_type='its'):
    predictions = np.array(predictions)

    if stds is None:
        stds = np.sqrt(np.var(predictions, axis=0))

    # Upper confidence bound (UCB) acquisition function
    if explore_type == 'ucb':
        explore_factor = 0.5
        mean = np.mean(predictions, axis=0)
        ucb = mean - explore_factor * stds
        sorted_indices = np.argsort(ucb)

    # Expected improvement (EI) acquisition function
    elif explore_type == 'ei':
        ei_calibration_factor = 5.
        mean = list(np.mean(predictions, axis=0))
        factored_stds = list(stds / ei_calibration_factor)
        min_y = ytrain.min()
        gam = [(min_y - mean[i]) / factored_stds[i] for i in range(len(mean))]
        ei = [-1 * factored_stds[i] * (gam[i] * norm.cdf(gam[i]) + norm.pdf(gam[i]))
              for i in range(len(mean))]
        sorted_indices = np.argsort(ei)

    # Probability of improvement (PI) acquisition function
    elif explore_type == 'pi':
        mean = list(np.mean(predictions, axis=0))
        stds = list(stds)
        min_y = ytrain.min()
        pi = [-1 * norm.cdf(min_y, loc=mean[i], scale=stds[i]) for i in range(len(mean))]
        sorted_indices = np.argsort(pi)

    # Thompson sampling (TS) acquisition function
    elif explore_type == 'ts':
        rand_ind = np.random.randint(predictions.shape[0])
        ts = predictions[rand_ind,:]
        sorted_indices = np.argsort(ts)

    # Top exploitation 
    elif explore_type == 'percentile':
        min_prediction = np.min(predictions, axis=0)
        sorted_indices = np.argsort(min_prediction)

    # Top mean
    elif explore_type == 'mean':
        mean = np.mean(predictions, axis=0)
        sorted_indices = np.argsort(mean)

    elif explore_type == 'confidence':
        confidence_factor = 2
        mean = np.mean(predictions, axis=0)
        conf = mean + confidence_factor * stds
        sorted_indices = np.argsort(conf)

    # Independent Thompson sampling (ITS) acquisition function
    elif explore_type == 'its':
        mean = np.mean(predictions, axis=0)
        samples = np.random.normal(mean, stds)
        sorted_indices = np.argsort(samples)

    else:
        print('{} is not a valid exploration type'.format(explore_type))
        raise NotImplementedError()

    return sorted_indices