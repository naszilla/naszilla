import numpy as np
import sys

# Different acquisition functions that can be used by BANANAS
def acq_fn(predictions, explore_type='its'):
    predictions = np.array(predictions)

    # Upper confidence bound (UCB) acquisition function
    if explore_type == 'ucb':
        explore_factor = 0.5
        mean = np.mean(predictions, axis=0)
        std = np.sqrt(np.var(predictions, axis=0))
        ucb = mean - explore_factor * std
        sorted_indices = np.argsort(ucb)

    # Expected improvement (EI) acquisition function
    elif explore_type == 'ei':
        ei_calibration_factor = 5.
        mean = list(np.mean(predictions, axis=0))
        std = list(np.sqrt(np.var(predictions, axis=0)) /
                   ei_calibration_factor)

        min_y = ytrain.min()
        gam = [(min_y - mean[i]) / std[i] for i in range(len(mean))]
        ei = [-1 * std[i] * (gam[i] * norm.cdf(gam[i]) + norm.pdf(gam[i]))
              for i in range(len(mean))]
        sorted_indices = np.argsort(ei)

    # Probability of improvement (PI) acquisition function
    elif explore_type == 'pi':
        mean = list(np.mean(predictions, axis=0))
        std = list(np.sqrt(np.var(predictions, axis=0)))
        min_y = ytrain.min()
        pi = [-1 * norm.cdf(min_y, loc=mean[i], scale=std[i]) for i in range(len(mean))]
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

    # Independent Thompson sampling (ITS) acquisition function
    elif explore_type == 'its':
        mean = np.mean(predictions, axis=0)
        std = np.sqrt(np.var(predictions, axis=0))
        samples = np.random.normal(mean, std)
        sorted_indices = np.argsort(samples)

    else:
        print('Invalid exploration type in meta neuralnet search', explore_type)
        sys.exit()

    return sorted_indices