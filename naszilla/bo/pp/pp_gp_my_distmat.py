"""
Classes for GP models without any PP backend, using a given distance matrix.
"""

from argparse import Namespace
import time
import copy
import numpy as np
from scipy.spatial.distance import cdist 

from naszilla.bo.pp.pp_core import DiscPP
from naszilla.bo.pp.gp.gp_utils import kern_exp_quad, kern_matern32, \
  get_cholesky_decomp, solve_upper_triangular, solve_lower_triangular, \
  sample_mvn, squared_euc_distmat, kern_distmat
from naszilla.bo.util.print_utils import suppress_stdout_stderr


class MyGpDistmatPP(DiscPP):
  """ GPs using a kernel specified by a given distance matrix, without any PP
      backend """

  def __init__(self, data=None, modelp=None, printFlag=True):
    """ Constructor """
    self.set_model_params(modelp)
    self.set_data(data)
    self.set_model()
    super(MyGpDistmatPP,self).__init__()
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
    """ Returns model object """
    return None

  def infer_post_and_update_samples(self, print_result=False):
    """ Update self.sample_list """
    self.sample_list = [Namespace(ls=self.modelp.kernp.ls,
                                  alpha=self.modelp.kernp.alpha,
                                  sigma=self.modelp.kernp.sigma)]
    if print_result: self.print_inference_result()

  def get_distmat(self, xmat1, xmat2):
    """ Get distance matrix """
    #return squared_euc_distmat(xmat1, xmat2, .5)
    search_space = self.modelp.search_space
    if search_space == 'nasbench_101':
        from nas_benchmarks import Nasbench101 as NB
    elif search_space == 'nasbench_201':
        from nas_benchmarks import Nasbench201 as NB
    elif search_space == 'nasbench_301':
        from nas_benchmarks import Nasbench301 as NB
        
    self.distmat = NB.generate_distance_matrix

    return self.distmat(xmat1, xmat2, self.modelp.distance)

  def print_inference_result(self):
    """ Print results of stan inference """
    print('*ls pt est = '+str(self.sample_list[0].ls)+'.')
    print('*alpha pt est = '+str(self.sample_list[0].alpha)+'.')
    print('*sigma pt est = '+str(self.sample_list[0].sigma)+'.')
    print('-----')

  def sample_pp_post_pred(self, nsamp, input_list, full_cov=False):
    """ Sample from posterior predictive of PP.
        Inputs:
          input_list - list of np arrays size=(-1,)
        Returns:
          list (len input_list) of np arrays (size=(nsamp,1))."""
    samp = self.sample_list[0]
    postmu, postcov = self.gp_post(self.data.X, self.data.y, input_list,
                                   samp.ls, samp.alpha, samp.sigma, full_cov)
    if full_cov:
      ppred_list = list(sample_mvn(postmu, postcov, nsamp))
    else:
      ppred_list = list(np.random.normal(postmu.reshape(-1,),
                                         postcov.reshape(-1,),
                                         size=(nsamp, len(input_list))))
    return list(np.stack(ppred_list).T), ppred_list

  def sample_pp_pred(self, nsamp, input_list, lv=None):
    """ Sample from predictive of PP for parameter lv.
        Returns: list (len input_list) of np arrays (size (nsamp,1))."""
    if lv is None:
      lv = self.sample_list[0]
    postmu, postcov = self.gp_post(self.data.X, self.data.y, input_list, lv.ls,
                                   lv.alpha, lv.sigma)
    pred_list = list(sample_mvn(postmu, postcov, 1)) ###TODO: sample from this mean nsamp times
    return list(np.stack(pred_list).T), pred_list

  def gp_post(self, x_train_list, y_train_arr, x_pred_list, ls, alpha, sigma,
              full_cov=True):
    """ Compute parameters of GP posterior """
    kernel = lambda a, b, c, d: kern_distmat(a, b, c, d, self.get_distmat)
    k11_nonoise = kernel(x_train_list, x_train_list, ls, alpha)
    lmat = get_cholesky_decomp(k11_nonoise, sigma, 'try_first')
    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat,
                                  y_train_arr))
    k21 = kernel(x_pred_list, x_train_list, ls, alpha)
    mu2 = k21.dot(smat)
    k22 = kernel(x_pred_list, x_pred_list, ls, alpha)
    vmat = solve_lower_triangular(lmat, k21.T)
    k2 = k22 - vmat.T.dot(vmat)
    if full_cov is False:
      k2 = np.sqrt(np.diag(k2))
    return mu2, k2

  # Utilities
  def print_str(self):
    """ Print a description string """
    print('*MyGpDistmatPP with modelp='+str(self.modelp)+'.')
    print('-----')
