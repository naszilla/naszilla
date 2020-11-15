"""
Classes for hierarchical GP models with George PP
"""

from argparse import Namespace
import numpy as np
import scipy.optimize as spo
import george
import emcee

from naszilla.bo.pp.pp_core import DiscPP

class GeorgeGpPP(DiscPP):
  """ Hierarchical GPs implemented with George """

  def __init__(self,data=None,modelp=None,printFlag=True):
    """ Constructor """
    self.set_data(data)
    self.set_model_params(modelp)
    self.ndimx = self.modelp.ndimx
    self.set_kernel()
    self.set_model()
    super(GeorgeGpPP,self).__init__()
    if printFlag:
      self.print_str()

  def set_data(self,data):
    if data is None:
      pass #TODO: handle case where there's no data
    self.data = data

  def set_model_params(self,modelp):
    if modelp is None:
      modelp = Namespace(ndimx=1, noiseVar=1e-3, kernLs=1.5, kernStr='mat',
        fitType='mle')
    self.modelp = modelp

  def set_kernel(self):
    """ Set kernel for GP """
    if self.modelp.kernStr=='mat':
      self.kernel = self.data.y.var() * \
        george.kernels.Matern52Kernel(self.modelp.kernLs, ndim=self.ndimx)
    if self.modelp.kernStr=='rbf': # NOTE: periodically produces errors
      self.kernel = self.data.y.var() * \
        george.kernels.ExpSquaredKernel(self.modelp.kernLs, ndim=self.ndimx)

  def set_model(self):
    """ Set GP regression model """
    self.model = self.get_model()
    self.model.compute(self.data.X)
    self.fit_hyperparams(printOut=False)

  def get_model(self):
    """ Returns GPRegression model """
    return george.GP(kernel=self.kernel,fit_mean=True)
  
  def fit_hyperparams(self,printOut=False):
    if self.modelp.fitType=='mle':
      spo.minimize(self.neg_log_like, self.model.get_parameter_vector(),
        jac=True)
    elif self.modelp.fitType=='bayes':
      self.nburnin = 200
      nsamp = 200
      nwalkers = 36
      gpdim = len(self.model)
      self.sampler = emcee.EnsembleSampler(nwalkers, gpdim, self.log_post)
      p0 = self.model.get_parameter_vector() + 1e-4*np.random.randn(nwalkers,
        gpdim)
      print 'Running burn-in.'
      p0, _, _ = self.sampler.run_mcmc(p0, self.nburnin)
      print 'Running main chain.'
      self.sampler.run_mcmc(p0, nsamp)
    if printOut:
      print 'Final GP hyperparam (in opt or MCMC chain):'
      print self.model.get_parameter_dict()

  def infer_post_and_update_samples(self):
    """ Update self.sample_list """
    self.sample_list = [None] #TODO: need to not-break ts fn in maker_bayesopt.py

  def sample_pp_post_pred(self,nsamp,input_list):
    """ Sample from posterior predictive of PP.
        Inputs:
          input_list - list of np arrays size=(-1,)
        Returns:
          list (len input_list) of np arrays (size=(nsamp,1))."""
    inputArray = np.array(input_list)
    if self.modelp.fitType=='mle':
      inputArray = np.array(input_list)
      ppredArray = self.model.sample_conditional(self.data.y.flatten(),
        inputArray, nsamp).T
    elif self.modelp.fitType=='bayes':
      ppredArray = np.zeros(shape=[len(input_list),nsamp])
      for s in range(nsamp):
        walkidx = np.random.randint(self.sampler.chain.shape[0])
        sampidx = np.random.randint(self.nburnin, self.sampler.chain.shape[1])
        hparamSamp = self.sampler.chain[walkidx, sampidx]
        print 'hparamSamp = ' + str(hparamSamp) # TODO: remove print statement
        self.model.set_parameter_vector(hparamSamp)
        ppredArray[:,s] = self.model.sample_conditional(self.data.y.flatten(),
          inputArray, 1).flatten()
    return list(ppredArray) # each element is row in ppredArray matrix

  def sample_pp_pred(self,nsamp,input_list,lv=None):
    """ Sample from predictive of PP for parameter lv.
        Returns: list (len input_list) of np arrays (size (nsamp,1))."""
    if self.modelp.fitType=='bayes':
      print('*WARNING: fitType=bayes not implemented for sample_pp_pred. \
        Reverting to fitType=mle')
      # TODO: Equivalent algo for fitType=='bayes':
      #   - draw posterior sample path over all xin in input_list
      #   - draw pred samples around sample path pt, based on noise model
    inputArray = np.array(input_list)
    samplePath = self.model.sample_conditional(self.data.y.flatten(),
      inputArray).reshape(-1,)
    return [np.random.normal(s,np.sqrt(self.modelp.noiseVar),nsamp).reshape(-1,)
      for s in samplePath]

  def neg_log_like(self,hparams):
    """ Compute and return the negative log likelihood for model
        hyperparameters hparams, as well as its gradient. """
    self.model.set_parameter_vector(hparams)
    g = self.model.grad_log_likelihood(self.data.y.flatten(), quiet=True)
    return -self.model.log_likelihood(self.data.y.flatten(), quiet=True), -g

  def log_post(self,hparams):
    """ Compute and return the log posterior density (up to constant of
        proportionality) for the model hyperparameters hparams. """
    # Uniform prior between -100 and 100, for each hyperparam
    if np.any((-100 > hparams[1:]) + (hparams[1:] > 100)):
      return -np.inf
    self.model.set_parameter_vector(hparams)
    return self.model.log_likelihood(self.data.y.flatten(), quiet=True)

  # Utilities
  def print_str(self):
    """ Print a description string """
    print '*GeorgeGpPP with modelp='+str(self.modelp)+'.'
    print '-----'
