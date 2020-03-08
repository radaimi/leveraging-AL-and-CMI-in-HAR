
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:05:31 2018
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sampling_def import SamplingMethod


"""Informative and diverse batch sampler that samples points with small margin
while maintaining same distribution over clusters as entire training data.
Batch is created by sorting datapoints by increasing margin and then growing
the batch greedily.  A point is added to the batch if the result batch still
respects the constraint that the cluster distribution of the batch will
match the cluster distribution of the entire training set.
"""



class InformativeClusterDiverseSampler(SamplingMethod):
  """Selects batch based on informative and diverse criteria.
    Returns highest uncertainty lowest margin points while maintaining
    same distribution over clusters as entire dataset.
  """

  def __init__(self, X, y, seed):
    self.name = 'informative_and_diverse'
    self.X = X
    self.flat_X = self.flatten_X()
    # y only used for determining how many clusters there should be
    # probably not practical to assume we know # of classes before hand
    # should also probably scale with dimensionality of data
    self.y = y
    self.n_clusters = len(list(set(y)))
    self.cluster_model = MiniBatchKMeans(n_clusters=self.n_clusters)
    self.cluster_data()

  def cluster_data(self):
    # Probably okay to always use MiniBatchKMeans
    # Should standardize data before clustering
    # Can cluster on standardized data but train on raw features if desired
    self.cluster_model.fit(self.flat_X)
    unique, counts = np.unique(self.cluster_model.labels_, return_counts=True)
    self.cluster_prob = counts/sum(counts)
    self.cluster_labels = self.cluster_model.labels_

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns a batch of size N using informative and diverse selection.
    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size
    Returns:
      indices of points selected to add using margin active learner
    """
    try:
      distances = model.decision_function(self.X)
    except:
      distances = model.predict_proba(self.X)
    if len(distances.shape) < 2:
      min_margin = abs(distances)
    else:
      sort_distances = np.sort(distances, 1)[:, -2:]
      min_margin = sort_distances[:, 1] - sort_distances[:, 0]
    rank_ind = np.argsort(min_margin)
    rank_ind = [i for i in rank_ind if i not in already_selected]
    new_batch_cluster_counts = [0 for _ in range(self.n_clusters)]
    new_batch = []
    for i in rank_ind:
      if len(new_batch) == N:
        break
      label = self.cluster_labels[i]
      if new_batch_cluster_counts[label] / N < self.cluster_prob[label]:
        new_batch.append(i)
        new_batch_cluster_counts[label] += 1
    n_slot_remaining = N - len(new_batch)
    batch_filler = list(set(rank_ind) - set(already_selected) - set(new_batch))
    new_batch.extend(batch_filler[0:n_slot_remaining])
    return new_batch, min_margin

  def to_dict(self):
    output = {}
    output['cluster_membership'] = self.cluster_labels
    return output