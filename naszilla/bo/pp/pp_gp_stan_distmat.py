"""
Classes for GP models with Stan, using a given distance matrix.
"""

from argparse import Namespace
import time
import copy
import numpy as np
from scipy.spatial.distance import cdist 
from bo.pp.pp_core import DiscPP
import bo.pp.stan.gp_distmat as gpstan
import bo.pp.stan.gp_distmat_fixedsig as gpstan_fixedsig
from bo.pp.gp.gp_utils import kern_exp_quad, kern_matern32, \
  get_cholesky_decomp, solve_upper_triangular, solve_lower_triangular, \
  sample_mvn, squared_euc_distmat, kern_distmat
from bo.util.print_utils import suppress_stdout_stderr

class StanGpDistmatPP(DiscPP):
  """ Hierarchical GPs using a given distance matrix, implemented with Stan """

  def __init__(self, data=None, modelp=None, printFlag=True):
    """ Constructor """
    self.set_model_params(modelp)
    self.set_data(data)
    self.ndimx = self.modelp.ndimx
    self.set_model()
    super(StanGpDistmatPP,self).__init__()
    if printFlag:
      self.print_str()

  def set_model_params(self, modelp):
    """ Set self.modelp """
    if modelp is None:
      pass #TODO
    self.modelp = modelp

  def set_data(self, data):
    """ Set self.data """
    if data is None:
      pass #TODO
    self.data_init = copy.deepcopy(data)
    self.data = copy.deepcopy(self.data_init)

  def set_model(self):
    """ Set GP regression model """
    self.model = self.get_model()

  def get_model(self):
    """ Returns GPRegression model """
    if self.modelp.model_str=='optfixedsig' or \
      self.modelp.model_str=='sampfixedsig':
      return gpstan_fixedsig.get_model(print_status=True)
    elif self.modelp.model_str=='opt' or self.modelp.model_str=='samp':
      return gpstan.get_model(print_status=True)
    elif self.modelp.model_str=='fixedparam':
      return None

  def infer_post_and_update_samples(self, seed=543210, print_result=False):
    """ Update self.sample_list """
    data_dict = self.get_stan_data_dict()
    with suppress_stdout_stderr():
      if self.modelp.model_str=='optfixedsig' or self.modelp.model_str=='opt':
        stanout = self.model.optimizing(data_dict, iter=self.modelp.infp.niter,
          #seed=seed, as_vector=True, algorithm='Newton')
          seed=seed, as_vector=True, algorithm='LBFGS')
      elif self.modelp.model_str=='samp' or self.modelp.model_str=='sampfixedsig':
        stanout = self.model.sampling(data_dict, iter=self.modelp.infp.niter +
          self.modelp.infp.nwarmup, warmup=self.modelp.infp.nwarmup, chains=1,
          seed=seed, refresh=1000)
      elif self.modelp.model_str=='fixedparam':
        stanout = None
      print('-----')
    self.sample_list = self.get_sample_list_from_stan_out(stanout)
    if print_result: self.print_inference_result()

  def get_stan_data_dict(self):
    """ Return data dict for stan sampling method """
    if self.modelp.model_str=='optfixedsig' or \
      self.modelp.model_str=='sampfixedsig':
      return {'ig1':self.modelp.kernp.ig1, 'ig2':self.modelp.kernp.ig2,
              'n1':self.modelp.kernp.n1, 'n2':self.modelp.kernp.n2,
              'sigma':self.modelp.kernp.sigma, 'D':self.ndimx,
              'N':len(self.data.X), 'y':self.data.y.flatten(),
              'distmat':self.get_distmat(self.data.X, self.data.X)}
    elif self.modelp.model_str=='opt' or self.modelp.model_str=='samp':
      return {'ig1':self.modelp.kernp.ig1, 'ig2':self.modelp.kernp.ig2,
              'n1':self.modelp.kernp.n1, 'n2':self.modelp.kernp.n2,
              'n3':self.modelp.kernp.n3, 'n4':self.modelp.kernp.n4,
              'D':self.ndimx, 'N':len(self.data.X), 'y':self.data.y.flatten(),
              'distmat':self.get_distmat(self.data.X, self.data.X)}

  def get_distmat(self, xmat1, xmat2):
    """ Get distance matrix """
    # For now, will compute squared euc distance * .5, on self.data.X
    return squared_euc_distmat(xmat1, xmat2, .5)

  def get_sample_list_from_stan_out(self, stanout):
    """ Convert stan output to sample_list """
    if self.modelp.model_str=='optfixedsig':
      return [Namespace(ls=stanout['rho'], alpha=stanout['alpha'],
        sigma=self.modelp.kernp.sigma)]
    elif self.modelp.model_str=='opt':
      return [Namespace(ls=stanout['rho'], alpha=stanout['alpha'],
        sigma=stanout['sigma'])]
    elif self.modelp.model_str=='sampfixedsig':
      sdict = stanout.extract(['rho','alpha'])
      return [Namespace(ls=sdict['rho'][i], alpha=sdict['alpha'][i],
        sigma=self.modelp.kernp.sigma) for i in range(sdict['rho'].shape[0])]
    elif self.modelp.model_str=='samp':
      sdict = stanout.extract(['rho','alpha','sigma'])
      return [Namespace(ls=sdict['rho'][i], alpha=sdict['alpha'][i],
        sigma=sdict['sigma'][i]) for i in range(sdict['rho'].shape[0])]
    elif self.modelp.model_str=='fixedparam':
      return [Namespace(ls=self.modelp.kernp.ls, alpha=self.modelp.kernp.alpha,
        sigma=self.modelp.kernp.sigma)]

  def print_inference_result(self):
    """ Print results of stan inference """
    if self.modelp.model_str=='optfixedsig' or self.modelp.model_str=='opt' or \
      self.modelp.model_str=='fixedparam':
      print('*ls pt est = '+str(self.sample_list[0].ls)+'.')
      print('*alpha pt est = '+str(self.sample_list[0].alpha)+'.')
      print('*sigma pt est = '+str(self.sample_list[0].sigma)+'.')
    elif self.modelp.model_str=='samp' or \
      self.modelp.model_str=='sampfixedsig':
      ls_arr = np.array([ns.ls for ns in self.sample_list])
      alpha_arr = np.array([ns.alpha for ns in self.sample_list])
      sigma_arr = np.array([ns.sigma for ns in self.sample_list])
      print('*ls mean = '+str(ls_arr.mean())+'.')
      print('*ls std = '+str(ls_arr.std())+'.')
      print('*alpha mean = '+str(alpha_arr.mean())+'.')
      print('*alpha std = '+str(alpha_arr.std())+'.')
      print('*sigma mean = '+str(sigma_arr.mean())+'.')
      print('*sigma std = '+str(sigma_arr.std())+'.')
    print('-----')

  def sample_pp_post_pred(self, nsamp, input_list, full_cov=False, nloop=None):
    """ Sample from posterior predictive of PP.
        Inputs:
          input_list - list of np arrays size=(-1,)
        Returns:
          list (len input_list) of np arrays (size=(nsamp,1))."""
    if self.modelp.model_str=='optfixedsig' or self.modelp.model_str=='opt' or \
        self.modelp.model_str=='fixedparam':
      nloop = 1
      sampids = [0]
    elif self.modelp.model_str=='samp' or \
      self.modelp.model_str=='sampfixedsig':
      if nloop is None: nloop=nsamp
      nsamp = int(nsamp/nloop)
      sampids = np.random.randint(len(self.sample_list), size=(nloop,))
    ppred_list = []
    for i in range(nloop):
      samp = self.sample_list[sampids[i]]
      postmu, postcov = self.gp_post(self.data.X, self.data.y,
        np.stack(input_list), samp.ls, samp.alpha, samp.sigma, full_cov)
      if full_cov:
        ppred_list.extend(list(sample_mvn(postmu, postcov, nsamp)))
      else:
        ppred_list.extend(list(np.random.normal(postmu.reshape(-1,),
          postcov.reshape(-1,), size=(nsamp, len(input_list)))))
    return list(np.stack(ppred_list).T), ppred_list

  def sample_pp_pred(self, nsamp, input_list, lv=None):
    """ Sample from predictive of PP for parameter lv.
        Returns: list (len input_list) of np arrays (size (nsamp,1))."""
    x_pred = np.stack(input_list)
    if lv is None:
      if self.modelp.model_str=='optfixedsig' or self.modelp.model_str=='opt' \
        or self.modelp.model_str=='fixedparam':
        lv = self.sample_list[0]
      elif self.modelp.model_str=='samp' or \
        self.modelp.model_str=='sampfixedsig':
        lv = self.sample_list[np.random.randint(len(self.sample_list))]
    postmu, postcov = self.gp_post(self.data.X, self.data.y, x_pred, lv.ls,
      lv.alpha, lv.sigma)
    pred_list = list(sample_mvn(postmu, postcov, 1)) ###TODO: sample from this mean nsamp times
    return list(np.stack(pred_list).T), pred_list

  def gp_post(self, x_train, y_train, x_pred, ls, alpha, sigma, full_cov=True):
    """ Compute parameters of GP posterior """
    kernel = lambda a, b, c, d: kern_distmat(a, b, c, d, self.get_distmat)
    k11_nonoise = kernel(x_train, x_train, ls, alpha)
    lmat = get_cholesky_decomp(k11_nonoise, sigma, 'try_first')
    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat, y_train))
    k21 = kernel(x_pred, x_train, ls, alpha)
    mu2 = k21.dot(smat)
    k22 = kernel(x_pred, x_pred, ls, alpha)
    vmat = solve_lower_triangular(lmat, k21.T)
    k2 = k22 - vmat.T.dot(vmat)
    if full_cov is False:
      k2 = np.sqrt(np.diag(k2))
    return mu2, k2

  # Utilities
  def print_str(self):
    """ Print a description string """
    print('*StanGpDistmatPP with modelp='+str(self.modelp)+'.')
    print('-----')
