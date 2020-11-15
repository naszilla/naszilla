"""
Make a point in a domain, and serialize it.
"""

import sys
import os
from argparse import Namespace, ArgumentParser
import pickle
import time
import numpy as np

from naszilla.bo.dom.real import RealDomain
from naszilla.bo.dom.list import ListDomain
from naszilla.bo.acq.acqmap import AcqMapper
from naszilla.bo.acq.acqopt import AcqOptimizer

def main(args, search_space, printinfo=False):
  starttime = time.time()
  
  # Load config and data
  makerp = pickle.load(open(args.configpkl, 'rb'))
  data = pickle.load(open(args.datapkl, 'rb'))

  if hasattr(args, 'mode') and args.mode == 'single_process':
    makerp.domp.mode = args.mode
    makerp.domp.iteridx = args.iteridx
    makerp.amp.modelp.mode = args.mode
  else:
    np.random.seed(args.seed)
  # Instantiate Domain, AcqMapper, AcqOptimizer
  dom = get_domain(makerp.domp, search_space)
  am = AcqMapper(data, makerp.amp, False)
  ao = AcqOptimizer(makerp.optp, False)
  # Optimize over domain to get nextpt 
  nextpt = ao.optimize(dom, am)
  # Serialize nextpt
  with open(args.nextptpkl, 'wb') as f:
    pickle.dump(nextpt, f)
  # Print
  itertime = time.time()-starttime
  if printinfo: print_info(nextpt, itertime, args.nextptpkl)

def get_domain(domp, search_space):
  """ Return Domain object. """
  if not hasattr(domp, 'dom_str'):
    domp.dom_str = 'real'
  if domp.dom_str=='real':
    return RealDomain(domp, False)
  elif domp.dom_str=='list':
    return ListDomain(search_space, domp, False)

def print_info(nextpt, itertime, nextptpkl):
  print('*Found nextpt = ' + str(nextpt) + '.')
  print('*Saved nextpt as ' + nextptpkl + '.')
  print('*Timing: makept took ' + str(itertime) + ' seconds.')
  print('-----')

if __name__ == "__main__":
  parser = ArgumentParser(description='Args for a single instance of acquisition optimization.')
  parser.add_argument('--seed', dest='seed', type=int, default=1111)
  parser.add_argument('--configpkl', dest='configpkl', type=str, default='config.pkl')
  parser.add_argument('--datapkl', dest='datapkl', type=str, default='data.pkl')
  parser.add_argument('--nextptpkl', dest='nextptpkl', type=str, default='nextpt.pkl')
  args = parser.parse_args()
  main(args, printinfo=False)
