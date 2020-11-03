"""
Utilities for Gaussian process (GP) inference
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist 
#import GPy as gpy


def kern_gibbscontext(xmatcon1, xmatcon2, xmatact1, xmatact2, theta, alpha,
  lscon, whichlsfn=1):
  """ Gibbs kernel (ls_fn of context only) """
  actdim = xmatact1.shape[1]
  lsarr1 = ls_fn(xmatcon1, theta, whichlsfn).flatten()
  lsarr2 = ls_fn(xmatcon2, theta, whichlsfn).flatten()
  sum_sq_ls = np.add.outer(lsarr1, lsarr2)
  inexp = -1. * np.divide(cdist(xmatact1, xmatact2, 'sqeuclidean'), sum_sq_ls)
  prod_ls = np.outer(lsarr1, lsarr2)
  #coef = np.power(np.divide(2*prod_ls, sum_sq_ls), actdim/2.) # Correct
  coef = 1.
  kern_gibbscontext_only_ns = np.multiply(coef, np.exp(inexp))
  kern_expquad_ns = kern_exp_quad_noscale(xmatcon1, xmatcon2, lscon)
  return alpha**2 * np.multiply(kern_gibbscontext_only_ns, kern_expquad_ns)

def kern_gibbs1d(xmat1, xmat2, theta, alpha):
  """ Gibbs kernel in 1d """
  lsarr1 = ls_fn(xmat1, theta).flatten()
  lsarr2 = ls_fn(xmat2, theta).flatten()
  sum_sq_ls = np.add.outer(lsarr1, lsarr2)
  prod_ls = np.outer(lsarr1, lsarr2) #TODO product of this for each dim
  coef = np.sqrt(np.divide(2*prod_ls, sum_sq_ls))
  inexp = cdist(xmat1, xmat2, 'sqeuclidean') / sum_sq_ls #TODO sum of this for each dim
  return alpha**2 * coef * np.exp(-1 * inexp)

def ls_fn(xmat, theta, whichlsfn=1):
  theta = np.array(theta).reshape(-1,1)
  if theta.shape[0]==2:
    if whichlsfn==1 or whichlsfn==2:
      return np.log(1 + np.exp(theta[0][0] + np.matmul(xmat,theta[1])))   # softplus transform
    elif whichlsfn==3:
      return np.exp(theta[0][0] + np.matmul(xmat,theta[1]))               # exp transform
  elif theta.shape[0]==3:
    if whichlsfn==1:
      return np.log(1 + np.exp(theta[0][0] + np.matmul(xmat,theta[1]) +
        np.matmul(np.power(xmat,2),theta[2])))                            # softplus transform
    elif whichlsfn==2:
      return np.log(1 + np.exp(theta[0][0] + np.matmul(xmat,theta[1]) +
        np.matmul(np.abs(xmat),theta[2])))                                # softplus on abs transform
    elif whichlsfn==3:
      return np.exp(theta[0][0] + np.matmul(xmat,theta[1]) +
        np.matmul(np.power(xmat,2),theta[2]))                             # exp transform
  else:
    print('ERROR: theta parameter is incorrect.')

def kern_matern32(xmat1, xmat2, ls, alpha):
  """ Matern 3/2 kernel, currently using GPy """
  kern = gpy.kern.Matern32(input_dim=xmat1.shape[1], variance=alpha**2,
    lengthscale=ls)
  return kern.K(xmat1,xmat2)

def kern_exp_quad(xmat1, xmat2, ls, alpha):
  """ Exponentiated quadratic kernel function aka squared exponential kernel
      aka RBF kernel """
  return alpha**2 * kern_exp_quad_noscale(xmat1, xmat2, ls)

def kern_exp_quad_noscale(xmat1, xmat2, ls):
  """ Exponentiated quadratic kernel function aka squared exponential kernel
      aka RBF kernel, without scale parameter. """
  sq_norm = (-1/(2 * ls**2)) * cdist(xmat1, xmat2, 'sqeuclidean')
  return np.exp(sq_norm)

def squared_euc_distmat(xmat1, xmat2, coef=1.):
  """ Distance matrix of squared euclidean distance (multiplied by coef)
      between points in xmat1 and xmat2. """
  return coef * cdist(xmat1, xmat2, 'sqeuclidean')

def kern_distmat(xmat1, xmat2, ls, alpha, distfn):
  """ Kernel for a given distmat, via passed-in distfn (which is assumed to be
      fn of xmat1 and xmat2 only) """
  distmat = distfn(xmat1, xmat2)
  sq_norm = -distmat / ls**2
  return alpha**2 * np.exp(sq_norm)

def get_cholesky_decomp(k11_nonoise, sigma, psd_str):
  """ Returns cholesky decomposition """
  if psd_str == 'try_first':
    k11 = k11_nonoise + sigma**2 * np.eye(k11_nonoise.shape[0])
    try:
      return stable_cholesky(k11, False)
    except np.linalg.linalg.LinAlgError:
      return get_cholesky_decomp(k11_nonoise, sigma, 'project_first')
  elif psd_str == 'project_first':
    k11_nonoise = project_symmetric_to_psd_cone(k11_nonoise)
    return get_cholesky_decomp(k11_nonoise, sigma, 'is_psd')
  elif psd_str == 'is_psd':
    k11 = k11_nonoise + sigma**2 * np.eye(k11_nonoise.shape[0])
    return stable_cholesky(k11)

def stable_cholesky(mmat, make_psd=True):
  """ Returns a 'stable' cholesky decomposition of mmat """
  if mmat.size == 0:
    return mmat
  try:
    lmat = np.linalg.cholesky(mmat)
  except np.linalg.linalg.LinAlgError as e:
    if not make_psd:
      raise e
    diag_noise_power = -11
    max_mmat = np.diag(mmat).max()
    diag_noise = np.diag(mmat).max() * 1e-11
    break_loop = False
    while not break_loop:
      try:
        lmat = np.linalg.cholesky(mmat + ((10**diag_noise_power) * max_mmat)  *
          np.eye(mmat.shape[0]))
        break_loop = True
      except np.linalg.linalg.LinAlgError:
        if diag_noise_power > -9:
          print('stable_cholesky failed with diag_noise_power=%d.'%(diag_noise_power))
        diag_noise_power += 1
      if diag_noise_power >= 5:
        print('***** stable_cholesky failed: added diag noise = %e'%(diag_noise))
  return lmat

def project_symmetric_to_psd_cone(mmat, is_symmetric=True, epsilon=0):
  """ Project symmetric matrix mmat to the PSD cone """
  if is_symmetric:
    try:
      eigvals, eigvecs = np.linalg.eigh(mmat)
    except np.linalg.LinAlgError:
      print('LinAlgError encountered with np.eigh. Defaulting to eig.')
      eigvals, eigvecs = np.linalg.eig(mmat)
      eigvals = np.real(eigvals)
      eigvecs = np.real(eigvecs)
  else:
    eigvals, eigvecs = np.linalg.eig(mmat)
  clipped_eigvals = np.clip(eigvals, epsilon, np.inf)
  return (eigvecs * clipped_eigvals).dot(eigvecs.T)

def solve_lower_triangular(amat, b):
  """ Solves amat*x=b when amat is lower triangular """
  return solve_triangular_base(amat, b, lower=True)

def solve_upper_triangular(amat, b):
  """ Solves amat*x=b when amat is upper triangular """
  return solve_triangular_base(amat, b, lower=False)

def solve_triangular_base(amat, b, lower):
  """ Solves amat*x=b when amat is a triangular matrix. """
  if amat.size == 0 and b.shape[0] == 0:
    return np.zeros((b.shape))
  else:
    return solve_triangular(amat, b, lower=lower)

def sample_mvn(mu, covmat, nsamp):
  """ Sample from multivariate normal distribution with mean mu and covariance
      matrix covmat """
  mu = mu.reshape(-1,)
  ndim = len(mu)
  lmat = stable_cholesky(covmat)
  umat = np.random.normal(size=(ndim, nsamp))
  return lmat.dot(umat).T + mu
