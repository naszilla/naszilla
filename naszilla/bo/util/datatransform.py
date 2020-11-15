"""
Classes for transforming data.
"""

from argparse import Namespace
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataTransformer(object):
  """ Class for transforming data """

  def __init__(self, datamat, printflag=True):
    """ Constructor
        Parameters:
          datamat - numpy array (n x d) of data to be transformed
    """
    self.datamat = datamat
    self.set_transformers()
    if printflag:
      self.print_str()

  def set_transformers(self):
    """ Set transformers using self.datamat """
    self.ss = StandardScaler()
    self.ss.fit(self.datamat)

  def transform_data(self, datamat=None):
    """ Return transformed datamat (default self.datamat) """
    if datamat is None:
      datamat = self.datamat
    return self.ss.transform(datamat)
 
  def inv_transform_data(self, datamat):
    """ Return inverse transform of datamat """
    return self.ss.inverse_transform(datamat)

  def print_str(self):
    """ Print a description string """
    print('*DataTransformer with self.datamat.shape = ' +
      str(self.datamat.shape) + '.')
    print('-----')
