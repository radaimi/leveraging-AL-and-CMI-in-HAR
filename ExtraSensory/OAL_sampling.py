# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 20:14:21 2018

@author: Rebecca Adaimi

OAL - sampling heuristic
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.stats import bernoulli
import numpy as np
from sampling_methods.sampling_def import SamplingMethod
from numpy.random import uniform


class OALSampler(SamplingMethod):
  """Selects batch based on informative and diverse criteria.

    Returns highest uncertainty lowest margin points while maintaining
    same distribution over clusters as entire dataset.
  """

  def __init__(self, X, y, seed, gamma):
    self.name = 'OAL'
    self.gamma = gamma
  def select_batch_(self, model,sample, N, **kwargs):


    distances = model.predict_proba(sample)
    if len(distances.shape) < 2:
      p = distances
    else:
      p = np.sort(distances, 1)[:, -1:]
      
    # print("P ",p)
    e = -1 * self.gamma * p
    ask_prob = np.exp(e)
    # print("ask_prob ",ask_prob)
    threshold = uniform(0,1)
    # print("threshold ",threshold)
#    z = bernoulli.rvs(ask_prob, size=1)
    return threshold < ask_prob

