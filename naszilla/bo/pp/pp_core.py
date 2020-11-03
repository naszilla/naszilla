"""
Base classes for probabilistic programs.
"""

import pickle

class DiscPP(object):
  """ Parent class for discriminative probabilistic programs """

  def __init__(self):
    """ Constructor """
    self.sample_list = []
    if not hasattr(self,'data'):
      raise NotImplementedError('Implement var data in a child class')
    #if not hasattr(self,'ndimx'):
      #raise NotImplementedError('Implement var ndimx in a child class')
    #if not hasattr(self,'ndataInit'):
      #raise NotImplementedError('Implement var ndataInit in a child class')

  def infer_post_and_update_samples(self,nsamp):
    """ Run an inference algorithm (given self.data), draw samples from the
        posterior, and store in self.sample_list. """
    raise NotImplementedError('Implement method in a child class')

  def sample_pp_post_pred(self,nsamp,input_list):
    """ Sample nsamp times from PP posterior predictive, for each x-input in
    input_list """
    raise NotImplementedError('Implement method in a child class')

  def sample_pp_pred(self,nsamp,input_list,lv_list=None):
    """ Sample nsamp times from PP predictive for parameter lv, for each
    x-input in input_list. If lv is None, draw it uniformly at random
    from self.sample_list. """
    raise NotImplementedError('Implement method in a child class')

  def add_new_data(self,newData):
    """ Add data (newData) to self.data """
    raise NotImplementedError('Implement method in a child class')

  def get_namespace_to_save(self):
    """ Return namespace containing object info (to save to file) """
    raise NotImplementedError('Implement method in a child class')

  def save_namespace_to_file(self,fileStr,printFlag):
    """ Saves results from get_namespace_to_save in fileStr """
    ppNamespaceToSave = self.get_namespace_to_save()
    ff = open(fileStr,'wb')
    pickle.dump(ppNamespaceToSave,ff)
    ff.close()
    if printFlag:
      print('*Saved DiscPP Namespace in pickle file: ' +fileStr+'\n-----')
