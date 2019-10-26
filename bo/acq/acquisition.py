"""
Classes to manage acquisition functions.
"""

from argparse import Namespace
import numpy as np
from scipy.stats import norm

class Acquisitioner(object):
  """ Class to manage acquisition functions """

  def __init__(self, data, acqp=None, print_flag=True):
    """ Constructor
        Parameters:
          acqp - Namespace of acquisition parameters
          print_flag - True or False
    """
    self.data = data
    self.set_acq_params(acqp)
    self.set_acq_method()
    if print_flag: self.print_str()

  def set_acq_params(self, acqp):
    """ Set the acquisition params.
        Parameters:
          acqp - Namespace of acquisition parameters """
    if acqp is None:
      acqp = Namespace(acq_str='ei', pmout_str='sample')
    self.acqp = acqp

  def set_acq_method(self):
    """ Set the acquisition method """
    if self.acqp.acq_str=='ei': self.acq_method = self.ei
    if self.acqp.acq_str=='pi': self.acq_method = self.pi
    if self.acqp.acq_str=='ts': self.acq_method = self.ts
    if self.acqp.acq_str=='ucb': self.acq_method = self.ucb
    if self.acqp.acq_str=='rand': self.acq_method = self.rand
    if self.acqp.acq_str=='null': self.acq_method = self.null
    #if self.acqp.acqStr=='map': return self.map

  def ei(self, pmout):
    """ Expected improvement (EI) """
    if self.acqp.pmout_str=='sample':
      return self.bbacq_ei(pmout)

  def pi(self, pmout):
    """ Probability of improvement (PI) """
    if self.acqp.pmout_str=='sample':
      return self.bbacq_pi(pmout)

  def ucb(self, pmout):
    """ Upper (lower) confidence bound (UCB) """
    if self.acqp.pmout_str=='sample':
      return self.bbacq_ucb(pmout)

  def ts(self, pmout):
    """ Thompson sampling (TS) """
    if self.acqp.pmout_str=='sample':
      return self.bbacq_ts(pmout)

  def rand(self, pmout):
    """ Uniform random sampling """
    return np.random.random()

  def null(self, pmout):
    """ Return constant 0. """
    return 0.

  # Black Box Acquisition Functions
  def bbacq_ei(self, pmout_samp, normal=False):
    """ Black box acquisition: BB-EI
        Input: pmout_samp: post-pred samples - np array (shape=(nsamp,1))
        Returns: EI acq value """
    youts = np.array(pmout_samp).flatten()
    nsamp = youts.shape[0]
    if normal:
      mu = np.mean(youts)
      sig = np.std(youts)
      gam = (self.data.y.min() - mu) / sig
      eiVal = -1*sig*(gam*norm.cdf(gam) + norm.pdf(gam))
    else:
      diffs = self.data.y.min() - youts
      ind_below_min = np.argwhere(diffs>0)
      eiVal = -1*np.sum(diffs[ind_below_min])/float(nsamp) if \
        len(ind_below_min)>0 else 0
    return eiVal

  def bbacq_pi(self, pmout_samp, normal=False):
    """ Black box acquisition: BB-PI
        Input: pmout_samp: post-pred samples - np array (shape=(nsamp,1))
        Returns: PI acq value """
    youts = np.array(pmout_samp).flatten()
    nsamp = youts.shape[0]
    if normal:
      mu = np.mean(youts)
      sig = np.sqrt(np.var(youts))
      piVal = -1*norm.cdf(self.data.y.min(),loc=mu,scale=sig)
    else:
      piVal = -1*len(np.argwhere(youts<self.data.y.min()))/float(nsamp)
    return piVal

  def bbacq_ucb(self, pmout_samp, beta=0.5, normal=True):
    """ Black box acquisition: BB-UCB
        Input: pmout_samp: post-pred samples - np array (shape=(nsamp,1))
        Returns: UCB acq value """
    youts = np.array(pmout_samp).flatten()
    nsamp = youts.shape[0]
    if normal:
      ucbVal = np.mean(youts) - beta*np.sqrt(np.var(youts))
    else:
      # TODO replace below with nonparametric ucb estimate
      ucbVal = np.mean(youts) - beta*np.sqrt(np.var(youts))
    return ucbVal

  def bbacq_ts(self, pmout_samp):
    """ Black box acquisition: BB-TS
        Input: pmout_samp: post-pred samples - np array (shape=(nsamp,1))
        Returns: TS acq value """
    return pmout_samp.mean()

  # Utilities
  def print_str(self):
    """ print a description string """
    print('*Acquisitioner with acqp='+str(self.acqp)+'.')
    print('-----')
