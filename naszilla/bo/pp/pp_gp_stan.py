"""
Classes for GP models with Stan
"""

from argparse import Namespace
import time
import numpy as np
import copy

from naszilla.bo.pp.pp_core import DiscPP
import naszilla.bo.pp.stan.gp_hier2 as gpstan2
import naszilla.bo.pp.stan.gp_hier3 as gpstan3
import naszilla.bo.pp.stan.gp_hier2_matern as gpstan2_matern
from naszilla.bo.pp.gp.gp_utils import kern_exp_quad, kern_matern32, \
  get_cholesky_decomp, solve_upper_triangular, solve_lower_triangular, \
  sample_mvn
from naszilla.bo.util.print_utils import suppress_stdout_stderr

class StanGpPP(DiscPP):
  """ Hierarchical GPs implemented with Stan """

  def __init__(self, data=None, modelp=None, printFlag=True):
    """ Constructor """
    self.set_model_params(modelp)
    self.set_data(data)
    self.ndimx = self.modelp.ndimx
    self.set_model()
    super(StanGpPP,self).__init__()
    if printFlag:
      self.print_str()

  def set_model_params(self,modelp):
    if modelp is None:
      modelp = Namespace(ndimx=1, model_str='optfixedsig',
        gp_mean_transf_str='constant')
      if modelp.model_str=='optfixedsig':
        modelp.kernp = Namespace(u1=.1, u2=5., n1=10., n2=10., sigma=1e-5)
        modelp.infp = Namespace(niter=1000)
      elif modelp.model_str=='opt' or modelp.model_str=='optmatern32':
        modelp.kernp = Namespace(ig1=1., ig2=5., n1=10., n2=20., n3=.01,
          n4=.01)
        modelp.infp = Namespace(niter=1000)
      elif modelp.model_str=='samp' or modelp.model_str=='sampmatern32':
        modelp.kernp = Namespace(ig1=1., ig2=5., n1=10., n2=20., n3=.01,
          n4=.01)
        modelp.infp = Namespace(niter=1500, nwarmup=500)
    self.modelp = modelp

  def set_data(self, data):
    """ Set self.data """
    if data is None:
      pass #TODO: handle case where there's no data
    self.data_init = copy.deepcopy(data)
    self.data = self.get_transformed_data(self.data_init,
      self.modelp.gp_mean_transf_str)

  def get_transformed_data(self, data, transf_str='linear'):
    """ Transform data, for non-zero-mean GP """
    newdata = Namespace(X=data.X)
    if transf_str=='linear':
      mmat,_,_,_ = np.linalg.lstsq(np.concatenate([data.X,
        np.ones((data.X.shape[0],1))],1), data.y.flatten(), rcond=None)
      self.gp_mean_vec = lambda x: np.matmul(np.concatenate([x,
        np.ones((x.shape[0],1))],1), mmat)
      newdata.y = data.y - self.gp_mean_vec(data.X).reshape(-1,1)
    if transf_str=='constant':
      yconstant = data.y.mean()
      #yconstant = 0. 
      self.gp_mean_vec = lambda x: np.array([yconstant for xcomp in x])
      newdata.y = data.y - self.gp_mean_vec(data.X).reshape(-1,1)
    return newdata

  def set_model(self):
    """ Set GP regression model """
    self.model = self.get_model()

  def get_model(self):
    """ Returns GPRegression model """
    if self.modelp.model_str=='optfixedsig':
      return gpstan3.get_model(print_status=False)
    elif self.modelp.model_str=='opt' or self.modelp.model_str=='samp':
      return gpstan2.get_model(print_status=False)
    elif self.modelp.model_str=='optmatern32' or \
      self.modelp.model_str=='sampmatern32':
      return gpstan2_matern.get_model(print_status=False)

  def infer_post_and_update_samples(self, seed=5000012, print_result=False):
    """ Update self.sample_list """
    data_dict = self.get_stan_data_dict()
    with suppress_stdout_stderr():
      if self.modelp.model_str=='optfixedsig' or self.modelp.model_str=='opt' \
        or self.modelp.model_str=='optmatern32':
        stanout = self.model.optimizing(data_dict, iter=self.modelp.infp.niter,
          #seed=seed, as_vector=True, algorithm='Newton')
          seed=seed, as_vector=True, algorithm='LBFGS')
      elif self.modelp.model_str=='samp' or self.modelp.model_str=='sampmatern32':
        stanout = self.model.sampling(data_dict, iter=self.modelp.infp.niter +
          self.modelp.infp.nwarmup, warmup=self.modelp.infp.nwarmup, chains=1,
          seed=seed, refresh=1000)
      print('-----')
    self.sample_list = self.get_sample_list_from_stan_out(stanout)
    if print_result: self.print_inference_result()

  def get_stan_data_dict(self):
    """ Return data dict for stan sampling method """
    if self.modelp.model_str=='optfixedsig':
      return {'u1':self.modelp.kernp.u1, 'u2':self.modelp.kernp.u2,
              'n1':self.modelp.kernp.n1, 'n2':self.modelp.kernp.n2,
              'sigma':self.modelp.kernp.sigma, 'D':self.ndimx,
              'N':len(self.data.X), 'x':self.data.X, 'y':self.data.y.flatten()}
    elif self.modelp.model_str=='opt' or self.modelp.model_str=='samp':
      return {'ig1':self.modelp.kernp.ig1, 'ig2':self.modelp.kernp.ig2,
              'n1':self.modelp.kernp.n1, 'n2':self.modelp.kernp.n2,
              'n3':self.modelp.kernp.n3, 'n4':self.modelp.kernp.n4,
              'D':self.ndimx, 'N':len(self.data.X), 'x':self.data.X,
              'y':self.data.y.flatten()}
    elif self.modelp.model_str=='optmatern32' or \
      self.modelp.model_str=='sampmatern32':
      return {'ig1':self.modelp.kernp.ig1, 'ig2':self.modelp.kernp.ig2,
              'n1':self.modelp.kernp.n1, 'n2':self.modelp.kernp.n2,
              'n3':self.modelp.kernp.n3, 'n4':self.modelp.kernp.n4,
              'D':self.ndimx, 'N':len(self.data.X), 'x':self.data.X,
              'y':self.data.y.flatten(), 'covid':2}

  def get_sample_list_from_stan_out(self, stanout):
    """ Convert stan output to sample_list """
    if self.modelp.model_str=='optfixedsig':
      return [Namespace(ls=stanout['rho'], alpha=stanout['alpha'],
        sigma=self.modelp.kernp.sigma)]
    elif self.modelp.model_str=='opt' or self.modelp.model_str=='optmatern32':
      return [Namespace(ls=stanout['rho'], alpha=stanout['alpha'],
        sigma=stanout['sigma'])]
    elif self.modelp.model_str=='samp' or \
      self.modelp.model_str=='sampmatern32':
      sdict = stanout.extract(['rho','alpha','sigma'])
      return [Namespace(ls=sdict['rho'][i], alpha=sdict['alpha'][i],
        sigma=sdict['sigma'][i]) for i in range(sdict['rho'].shape[0])]

  def print_inference_result(self):
    """ Print results of stan inference """
    if self.modelp.model_str=='optfixedsig' or self.modelp.model_str=='opt' or \
      self.modelp.model_str=='optmatern32':
      print('*ls pt est = '+str(self.sample_list[0].ls)+'.')
      print('*alpha pt est = '+str(self.sample_list[0].alpha)+'.')
      print('*sigma pt est = '+str(self.sample_list[0].sigma)+'.')
    elif self.modelp.model_str=='samp' or \
      self.modelp.model_str=='sampmatern32':
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
      self.modelp.model_str=='optmatern32':
      nloop = 1
      sampids = [0]
    elif self.modelp.model_str=='samp' or \
      self.modelp.model_str=='sampmatern32':
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
    return self.get_reverse_transform(list(np.stack(ppred_list).T), ppred_list,
      input_list)

  def sample_pp_pred(self, nsamp, input_list, lv=None):
    """ Sample from predictive of PP for parameter lv.
        Returns: list (len input_list) of np arrays (size (nsamp,1))."""
    x_pred = np.stack(input_list)
    if lv is None:
      if self.modelp.model_str=='optfixedsig' or self.modelp.model_str=='opt' \
        or self.modelp.model_str=='optmatern32':
        lv = self.sample_list[0]
      elif self.modelp.model_str=='samp' or \
        self.modelp.model_str=='sampmatern32':
        lv = self.sample_list[np.random.randint(len(self.sample_list))]
    postmu, postcov = self.gp_post(self.data.X, self.data.y, x_pred, lv.ls,
      lv.alpha, lv.sigma)
    pred_list = list(sample_mvn(postmu, postcov, 1)) ###TODO: sample from this mean nsamp times
    return self.get_reverse_transform(list(np.stack(pred_list).T), pred_list,
      input_list)

  def get_reverse_transform(self, pp1, pp2, input_list):
    """ Apply reverse of data transform to ppred or pred """
    pp1 = [pp1[i] + self.gp_mean_vec(input_list[i].reshape(1,-1)) for i in
           range(len(input_list))]
    pp2 = [psamp + self.gp_mean_vec(np.array(input_list)) for psamp in pp2]
    return pp1, pp2

  def gp_post(self, x_train, y_train, x_pred, ls, alpha, sigma, full_cov=True):
    """ Compute parameters of GP posterior """
    if self.modelp.model_str=='optmatern32' or \
      self.modelp.model_str=='sampmatern32':
      kernel = kern_matern32
    else:
      kernel = kern_exp_quad
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
    print('*StanGpPP with modelp='+str(self.modelp)+'.')
    print('-----')
