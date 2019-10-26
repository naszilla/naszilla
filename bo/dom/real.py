"""
Classes for real coordinate space domains.
"""

from argparse import Namespace
import numpy as np

class RealDomain(object):
  """ Class for defining sets in real coordinate (Euclidean) space """

  def __init__(self, domp=None, printFlag=True):
    """ Constructor
        Parameters:
          domp - domain parameters Namespace
    """
    self.set_domain_params(domp)
    self.ndimx = self.domp.ndimx
    if printFlag:
      self.print_str()

  def set_domain_params(self, domp):
    """ Set self.domp Namespace """
    if domp is None:
      domp = Namespace()
      domp.ndimx = 1
      domp.min_max = [(0,1)]*domp.ndimx
    self.domp = domp

  def is_in_domain(self, pt):
    """ Check if pt is in domain, and return True or False """
    pt = np.array(pt).reshape(-1)
    if pt.shape[0] != self.ndimx:
      ret=False
    else:
      bool_list = [pt[i]>=self.domp.min_max[i][0] and
        pt[i]<=self.domp.min_max[i][1] for i in range(self.ndimx)]
      ret=False if False in bool_list else True
    return ret

  def unif_rand_sample(self, n=1):
    """ Draws a sample uniformly at random from domain """
    li = [np.random.uniform(mm[0], mm[1], n) for mm in self.domp.min_max]
    return np.array(li).T

  def print_str(self):
    """ Print a description string """
    print('*RealDomain with domp = ' + str(self.domp) + '.')
    print('-----')
