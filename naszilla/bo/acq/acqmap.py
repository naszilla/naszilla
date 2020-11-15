"""
Classes to manage acqmap (acquisition maps from xin to acquisition value).
"""

from argparse import Namespace
import numpy as np
import copy

from naszilla.bo.acq.acquisition import Acquisitioner
from naszilla.bo.util.datatransform import DataTransformer
from naszilla.bo.pp.pp_gp_my_distmat import MyGpDistmatPP

class AcqMapper(object):
  """ Class to manage acqmap (acquisition map). """

  def __init__(self, data, amp, print_flag=True):
    """ Constructor
        Parameters:
          amp - Namespace of acqmap params
          print_flag - True or False
    """
    self.data = data
    self.set_am_params(amp)
    #self.setup_acqmap()
    if print_flag: self.print_str()

  def set_am_params(self, amp):
    """ Set the acqmap params.
        Inputs:
          amp - Namespace of acqmap parameters """
    self.amp = amp

  def get_acqmap(self, xin_is_list=True):
    """ Return acqmap.
        Inputs: xin_is_list True if input to acqmap is a list of xin """
    # Potentially do acqmap setup here. Could include inference,
    # cachining/computing quantities, instantiating objects used in acqmap
    # definition. This becomes important when we do sequential opt of acqmaps.
    return self.acqmap_list if xin_is_list else self.acqmap_single

  def acqmap_list(self, xin_list):
    """ Acqmap defined on a list of xin. """

    def get_trans_data():
      """ Returns transformed data. """
      dt = DataTransformer(self.data.y, False)
      return Namespace(X=self.data.X, y=dt.transform_data(self.data.y))

    def apply_acq_to_pmlist(pmlist, acq_str, trans_data):
      """ Apply acquisition to pmlist. """
      acqp = Namespace(acq_str=acq_str, pmout_str='sample')
      acq = Acquisitioner(trans_data, acqp, False)
      acqfn = acq.acq_method
      return [acqfn(p) for p in pmlist]

    def georgegp_acqmap(acq_str):
      """ Acqmaps for GeorgeGpPP """
      trans_data = get_trans_data()
      pp = GeorgeGpPP(trans_data, self.amp.modelp, False)
      pmlist = pp.sample_pp_pred(self.amp.nppred, xin_list) if acq_str=='ts' \
        else pp.sample_pp_post_pred(self.amp.nppred, xin_list)
      return apply_acq_to_pmlist(pmlist, acq_str, trans_data)

    def stangp_acqmap(acq_str):
      """ Acqmaps for StanGpPP """
      trans_data = get_trans_data()
      pp = StanGpPP(trans_data, self.amp.modelp, False)
      pp.infer_post_and_update_samples(print_result=True)
      pmlist, _ = pp.sample_pp_pred(self.amp.nppred, xin_list) if acq_str=='ts' \
        else pp.sample_pp_post_pred(self.amp.nppred, xin_list, full_cov=True, \
        nloop=np.min([50,self.amp.nppred]))
      return apply_acq_to_pmlist(pmlist, acq_str, trans_data)

    def mygpdistmat_acqmap(acq_str):
      """ Acqmaps for MyGpDistmatPP """
      trans_data = get_trans_data()
      pp = MyGpDistmatPP(trans_data, self.amp.modelp, False)
      pp.infer_post_and_update_samples(print_result=False)
      pmlist, _ = pp.sample_pp_pred(self.amp.nppred, xin_list) if acq_str=='ts' \
        else pp.sample_pp_post_pred(self.amp.nppred, xin_list, full_cov=True)
      return apply_acq_to_pmlist(pmlist, acq_str, trans_data)

    # Mapping of am_str to acqmap
    if self.amp.am_str=='georgegp_ei':
      return georgegp_acqmap('ei')
    elif self.amp.am_str=='georgegp_pi':
      return georgegp_acqmap('pi')
    elif self.amp.am_str=='georgegp_ucb':
      return georgegp_acqmap('ucb')
    elif self.amp.am_str=='georgegp_ts':
      return georgegp_acqmap('ts')
    elif self.amp.am_str=='stangp_ei':
      return stangp_acqmap('ei')
    elif self.amp.am_str=='stangp_pi':
      return stangp_acqmap('pi')
    elif self.amp.am_str=='stangp_ucb':
      return stangp_acqmap('ucb')
    elif self.amp.am_str=='stangp_ts':
      return stangp_acqmap('ts')
    elif self.amp.am_str=='mygpdistmat_ei':
      return mygpdistmat_acqmap('ei')
    elif self.amp.am_str=='mygpdistmat_pi':
      return mygpdistmat_acqmap('pi')
    elif self.amp.am_str=='mygpdistmat_ucb':
      return mygpdistmat_acqmap('ucb')
    elif self.amp.am_str=='mygpdistmat_ts':
      return mygpdistmat_acqmap('ts')
    elif self.amp.am_str=='null':
      return [0. for xin in xin_list]

  def acqmap_single(self, xin):
    """ Acqmap defined on a single xin. Returns acqmap(xin) value, not list. """
    return self.acqmap_list([xin])[0]

  def print_str(self):
    """ Print a description string """
    print('*AcqMapper with amp='+str(self.amp)
      +'.\n-----')
