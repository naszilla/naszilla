"""
Classes for list (discrete set) domains.
"""

from argparse import Namespace
import numpy as np


class ListDomain(object):
  """ Class for defining sets defined by a list of elements """

  def __init__(self, search_space, domp=None, printFlag=True):
    """ Constructor
        Parameters:
          domp - domain parameters Namespace
    """
    self.set_domain_params(domp)
    self.search_space = search_space
    self.init_domain_list()
    if printFlag:
      self.print_str()

  def set_domain_params(self, domp):
    """ Set self.domp Namespace """
    self.domp = domp

  def init_domain_list(self):
    """ Initialize self.domain_list. """
    if self.domp.set_domain_list_auto:
      self.set_domain_list_auto()
    else:
      self.domain_list = None

  def set_domain_list_auto(self):
    self.domain_list = self.search_space.get_arch_list(self.domp.aux_file_path)

  def set_domain_list(self, domain_list):
    """ Set self.domain_list, containing elements of domain """
    self.domain_list = domain_list

  def is_in_domain(self, pt):
    """ Check if pt is in domain, and return True or False """
    return pt in self.domain_list

  def unif_rand_sample(self, n=1, replace=True):
    """ Draws a sample uniformly at random from domain, returns as a list of
        len n, with (default) or without replacement. """
    if replace:
      randind = np.random.randint(len(self.domain_list), size=n)
    else:
      randind = np.arange(min(n, len(self.domain_list)))
    return [self.domain_list[i] for i in randind]

  def print_str(self):
    """ Print a description string """
    print('*ListDomain with domp = ' + str(self.domp) + '.')
    print('-----')
