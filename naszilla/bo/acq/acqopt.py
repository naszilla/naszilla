"""
Classes to perform acquisition function optimization.
"""

from argparse import Namespace
import numpy as np

class AcqOptimizer(object):
  """ Class to perform acquisition function optimization """

  def __init__(self, optp=None, print_flag=True):
    """ Constructor
        Inputs:
          optp - Namespace of opt parameters
          print_flag - True or False
    """
    self.set_opt_params(optp)
    if print_flag: self.print_str()

  def set_opt_params(self, optp):
    """ Set the optimizer params.
        Inputs:
          acqp - Namespace of acquisition parameters """
    if optp is None:
      optp = Namespace(opt_str='rand', max_iter=1000)
    self.optp = optp

  def optimize(self, dom, am):
    """ Optimize acqfn(probmap(x)) over x in domain """
    if self.optp.opt_str=='rand':
      return self.optimize_rand(dom, am)

  def optimize_rand(self, dom, am):
    """ Optimize acqmap(x) over domain via random search """
    xin_list = dom.unif_rand_sample(self.optp.max_iter)
    amlist = am.acqmap_list(xin_list)
    return xin_list[np.argmin(amlist)]

  # Utilities 
  def print_str(self):
    """ print a description string """
    print('*AcqOptimizer with optp='+str(self.optp)
      +'.\n-----')
