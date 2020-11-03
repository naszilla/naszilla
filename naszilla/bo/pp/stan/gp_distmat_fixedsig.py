"""
Functions to define and compile PPs in Stan, for model:
hierarchical GP (prior on rho, alpha) and fixed sigma, using a given
distance matrix.
"""

import time
import pickle
import pystan

def get_model(recompile=False, print_status=True):
  model_file_str = 'bo/pp/stan/hide_model/gp_distmat_fixedsig.pkl'

  if recompile:
    starttime = time.time()
    model = pystan.StanModel(model_code=get_model_code())
    buildtime = time.time()-starttime
    with open(model_file_str,'wb') as f:
      pickle.dump(model, f)
    if print_status:
      print('*Time taken to compile = '+ str(buildtime) +' seconds.\n-----')
      print('*Model saved in file ' + model_file_str + '.\n-----')
  else:
    model = pickle.load(open(model_file_str,'rb'))
    if print_status:
      print('*Model loaded from file ' + model_file_str + '.\n-----')
  return model


def get_model_code():
  """ Parse modelp and return stan model code """
  return """
  data {
    int<lower=1> N;
    matrix[N, N] distmat;
    vector[N] y;
    real<lower=0> ig1;
    real<lower=0> ig2;
    real<lower=0> n1;
    real<lower=0> n2;
    real<lower=0> sigma;
  }

  parameters {
    real<lower=0> rho;
    real<lower=0> alpha;
  }

  model {
    matrix[N, N] cov = square(alpha) * exp(-distmat / square(rho))
                       + diag_matrix(rep_vector(square(sigma), N));
    matrix[N, N] L_cov = cholesky_decompose(cov);
    rho ~ inv_gamma(ig1, ig2);
    alpha ~ normal(n1, n2);
    y ~ multi_normal_cholesky(rep_vector(0, N), L_cov);
  }
  """

if __name__ == '__main__':
  get_model()
