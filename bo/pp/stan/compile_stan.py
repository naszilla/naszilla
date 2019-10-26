"""
Script to compile stan models
"""

#import pp_new.stan.gp_hier2 as gpstan
#import pp_new.stan.gp_hier3 as gpstan
#import pp_new.stan.gp_hier2_matern as gpstan
import pp_new.stan.gp_distmat as gpstan
#import pp_new.stan.gp_distmat_fixedsig as gpstan


# Recompile model and return it
model = gpstan.get_model(recompile=True)
