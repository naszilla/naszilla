"""
Classes for ProBO (probabilistic programming BO) using makept strategy.
"""

import time
from argparse import Namespace
import subprocess
import os
import pickle
import numpy as np

from naszilla.bo.fn.functionhandler import get_fh
from naszilla.bo.ds.makept import main

class ProBO(object):
  """ Class to carry out ProBO (probabilistic programming BO) """

  def __init__(self, fn, search_space, aux_file_path, data=None, probop=None, printFlag=True):
    """ Constructor
        Parameters:
          fn - Function to query (experiment)
          data - Initial dataset Namespace (with keys: X, y)
          probop - probo parameters Namespace
    """
    self.data = data
    self.search_space = search_space
    self.set_probo_params(probop)
    self.set_fh(fn)
    self.set_tmpdir()
    self.auxpkl = aux_file_path
    if printFlag:
      self.print_str()

  def set_probo_params(self, probop):
    """ Set ProBO parameters """
    self.probop = probop

  def set_fh(self, fn):
    """ Set function handler """
    self.fh = get_fh(fn, self.data, self.probop.fhp)

  def set_tmpdir(self):
    """ Set tmp directory and files """
    if not os.path.exists(self.probop.tmpdir):
      os.makedirs(self.probop.tmpdir)
    self.configpkl = os.path.join(self.probop.tmpdir, 'config.pkl')
    self.datapkl = os.path.join(self.probop.tmpdir, 'data.pkl')
    self.nextptpkl = os.path.join(self.probop.tmpdir, 'nextpt.pkl')

  def run_bo(self, verbose=1):
    """ Main BO loop. """
    # Serialize makerp 
    with open(self.configpkl, 'wb') as f:
      pickle.dump(self.probop.makerp, f)
    print('*Saved self.probop.makerp as ' + self.configpkl + '.\n-----')
    # Iterate
    for iteridx in range(self.probop.niter):
      starttime = time.time()
      # Serialize current data
      with open(self.datapkl, 'wb') as f:
        pickle.dump(self.data, f)

      if not hasattr(self.probop, 'mode') or self.probop.mode == 'subprocess':
        subseed = np.random.randint(111111)
        subprocess.call(['python3', 'bo/ds/makept.py', '--configpkl', self.configpkl,
                         '--datapkl', self.datapkl, '--nextptpkl',
                         self.nextptpkl, '--seed', str(subseed)])
      elif self.probop.mode == 'single_process':
        args = Namespace(configpkl=self.configpkl, datapkl=self.datapkl, nextptpkl=self.nextptpkl,
            mode=self.probop.mode, iteridx=iteridx)
        main(args, self.search_space)

      # Call fn on nextpt
      nextpt = pickle.load(open(self.nextptpkl, 'rb'))
      self.fh.call_fn_and_add_data(nextpt)

      if verbose and (iteridx % 10 == 0):
        print('Finished GP-BayesOpt query', iteridx)
      itertime = time.time()-starttime

      self.post_iteration()

  def print_iter_info(self, iteridx, itertime):
    """ Print information at end of an iteration. """
    print('*Last query results: xin = ' + str(self.data.X[-1]) +
          ', yout = ' + str(self.data.y[-1]) + '.')
    print('*Timing: iteration took ' + str(itertime) + ' seconds.')
    print('*Finished ProBO iter = ' + str(iteridx+1) +
          '.\n' + '==='*20)

  def print_str(self):
    """ print a description string """
    print('*ProBO (using makept) with probop='+str(self.probop)
          + '.\n-----')

  def post_iteration(self):
    pairs = [(self.data.X[i], self.data.y[i]) for i in range(len(self.data.y))]
    pairs.sort(key = lambda x:x[1])
    with open(self.auxpkl, 'wb') as f:
      pickle.dump(pairs, f)



