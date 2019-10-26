"""
Classes to handle functions.
"""

from argparse import Namespace
import numpy as np

def get_fh(fn, data=None, fhp=None, print_flag=True):
  """ Returns a function handler object """
  if fhp is None:
    fhp=Namespace(fhstr='basic', namestr='noname')
  # Return FH object
  if fhp.fhstr=='basic':
    return BasicFH(fn, data, fhp, print_flag)
  elif fhp.fhstr=='extrainfo':
    return ExtraInfoFH(fn, data, fhp, print_flag)
  elif fhp.fhstr=='nannn':
    return NanNNFH(fn, data, fhp, print_flag)
  elif fhp.fhstr=='replacenannn':
    return ReplaceNanNNFH(fn, data, fhp, print_flag)
  elif fhp.fhstr=='object':
    return ObjectFH(fn, data, fhp, print_flag)


class BasicFH(object):
  """ Class to handle basic functions, which map from an array xin to a real
      value yout. """

  def __init__(self, fn, data=None, fhp=None, print_flag=True):
    """ Constructor.
        Inputs:
          pmp - Namespace of probmap params
          print_flag - True or False
    """
    self.fn = fn
    self.data = data
    self.fhp = fhp
    if print_flag: self.print_str()

  def call_fn_and_add_data(self, xin):
    """ Call self.fn(xin), and update self.data """
    yout = self.fn(xin)
    print('new datapoint score', yout)
    self.add_data_single(xin, yout)

  def add_data_single(self, xin, yout):
    """ Update self.data with a single xin yout pair.
        Inputs:
          xin: np.array size=(1, -1)
          yout: np.array size=(1, 1) """
    xin = np.array(xin).reshape(1, -1)
    yout = np.array(yout).reshape(1, 1)
    newdata = Namespace(X=xin, y=yout)
    self.add_data(newdata)

  def add_data(self, newdata):
    """ Update self.data with newdata Namespace.
        Inputs:
          newdata: Namespace with fields X and y """
    if self.data is None:
      self.data = newdata
    else:
      self.data.X = np.concatenate((self.data.X, newdata.X), 0)
      self.data.y = np.concatenate((self.data.y, newdata.y), 0)

  def print_str(self):
    """ Print a description string. """
    print('*BasicFH with fhp='+str(self.fhp)
      +'.\n-----')


class ExtraInfoFH(BasicFH):
  """ Class to handle functions that map from an array xin to a real
      value yout, but also return extra info """

  def __init__(self, fn, data=None, fhp=None, print_flag=True):
    """ Constructor.
        Inputs:
          pmp - Namespace of probmap params
          print_flag - True or False
    """
    super(ExtraInfoFH, self).__init__(fn, data, fhp, False)
    self.extrainfo = []
    if print_flag: self.print_str()

  def call_fn_and_add_data(self, xin):
    """ Call self.fn(xin), and update self.data """
    yout, exinf = self.fn(xin)
    self.add_data_single(xin, yout)
    self.extrainfo.append(exinf)

  def print_str(self):
    """ Print a description string. """
    print('*ExtraInfoFH with fhp='+str(self.fhp)
      +'.\n-----')


class NanNNFH(BasicFH):
  """ Class to handle NN functions that map from an array xin to either
      a real value yout or np.NaN, but also return extra info """

  def __init__(self, fn, data=None, fhp=None, print_flag=True):
    """ Constructor.
        Inputs:
          pmp - Namespace of probmap params
          print_flag - True or False
    """
    super(NanNNFH, self).__init__(fn, data, fhp, False)
    self.extrainfo = []
    if print_flag: self.print_str()

  def call_fn_and_add_data(self, xin):
    """ Call self.fn(xin), and update self.data """
    timethresh = 60.
    yout, walltime = self.fn(xin)
    if walltime > timethresh:
      self.add_data_single_nan(xin)
    else:
      self.add_data_single(xin, yout)
      self.possibly_init_xnan()
    exinf = Namespace(xin=xin, yout=yout, walltime=walltime)
    self.extrainfo.append(exinf)

  def add_data_single_nan(self, xin):
    """ Update self.data.X_nan with a single xin.
        Inputs:
          xin: np.array size=(1, -1) """
    xin = xin.reshape(1,-1)
    newdata = Namespace(X = np.ones((0, xin.shape[1])),
                        y = np.ones((0, 1)),
                        X_nan = xin)
    self.add_data_nan(newdata)

  def add_data_nan(self, newdata):
    """ Update self.data with newdata Namespace.
        Inputs:
          newdata: Namespace with fields X, y, X_nan """
    if self.data is None:
      self.data = newdata
    else:
      self.data.X_nan = np.concatenate((self.data.X_nan, newdata.X_nan), 0)

  def possibly_init_xnan(self):
    """ If self.data doesn't have X_nan, then create it. """
    if not hasattr(self.data, 'X_nan'):
      self.data.X_nan = np.ones((0, self.data.X.shape[1]))

  def print_str(self):
    """ Print a description string. """
    print('*NanNNFH with fhp='+str(self.fhp)
      +'.\n-----')


class ReplaceNanNNFH(BasicFH):
  """ Class to handle NN functions that map from an array xin to either
      a real value yout or np.NaN. If np.NaN, we replace it with a large
      positive value. We also return extra info """

  def __init__(self, fn, data=None, fhp=None, print_flag=True):
    """ Constructor.
        Inputs:
          pmp - Namespace of probmap params
          print_flag - True or False
    """
    super(ReplaceNanNNFH, self).__init__(fn, data, fhp, False)
    self.extrainfo = []
    if print_flag: self.print_str()

  def call_fn_and_add_data(self, xin):
    """ Call self.fn(xin), and update self.data """
    timethresh = 60.
    replace_nan_val = 5.
    yout, walltime = self.fn(xin)
    if walltime > timethresh:
      yout = replace_nan_val
    self.add_data_single(xin, yout)
    exinf = Namespace(xin=xin, yout=yout, walltime=walltime)
    self.extrainfo.append(exinf)

  def print_str(self):
    """ Print a description string. """
    print('*ReplaceNanNNFH with fhp='+str(self.fhp)
      +'.\n-----')


class ObjectFH(object):
  """ Class to handle basic functions, which map from some object xin to a real
      value yout. """

  def __init__(self, fn, data=None, fhp=None, print_flag=True):
    """ Constructor.
        Inputs:
          pmp - Namespace of probmap params
          print_flag - True or False
    """
    self.fn = fn
    self.data = data
    self.fhp = fhp
    if print_flag: self.print_str()

  def call_fn_and_add_data(self, xin):
    """ Call self.fn(xin), and update self.data """
    yout = self.fn(xin)
    self.add_data_single(xin, yout)

  def add_data_single(self, xin, yout):
    """ Update self.data with a single xin yout pair. """
    newdata = Namespace(X=[xin], y=np.array(yout).reshape(1, 1))
    self.add_data(newdata)

  def add_data(self, newdata):
    """ Update self.data with newdata Namespace.
        Inputs:
          newdata: Namespace with fields X and y """
    if self.data is None:
      self.data = newdata
    else:
      self.data.X.extend(newdata.X)
      self.data.y = np.concatenate((self.data.y, newdata.y), 0)

  def print_str(self):
    """ Print a description string. """
    print('*ObjectFH with fhp='+str(self.fhp)
      +'.\n-----')
