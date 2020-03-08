# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 12:22:40 2018

@author: Rebecca Adaimi

PAMAP2 Training and Testing OAL

"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import csv
from collections import Counter
from OAL_sampling import OALSampler
import random
from sklearn.metrics import classification_report, confusion_matrix
import copy

def classification_report_csv(subject, report):
  report_data = []
  lines = report.split('\n')

  for line in lines[2:-5]:
      row = {}
      row_data = line.split('      ')
      row['Participant'] = subject
      row['class'] = row_data[1]
      row['precision'] = float(row_data[2])
      row['recall'] = float(row_data[3])
      row['f1_score'] = float(row_data[4])
      row['support'] = float(row_data[5])
      report_data.append(row)
  for line in lines[-2:-1]:
      row = {}
      row_data = line.split('      ')
      row['Participant'] = np.nan
      row['class'] = row_data[0]
      row['precision'] = float(row_data[1])
      row['recall'] = float(row_data[2])
      row['f1_score'] = float(row_data[3])
      row['support'] = float(row_data[4])
      report_data.append(row)

  dataframe = pd.DataFrame.from_dict(report_data)
  return dataframe

def get_class_counts(y_full, y):
  """Gets the count of all classes in a sample.
  Args:
    y_full: full target vector containing all classes
    y: sample vector for which to perform the count
  Returns:
    count of classes for the sample vector y, the class order for count will
    be the same as long as same y_full is fed in
  """
  classes = np.unique(y_full)
  classes = np.sort(classes)
  unique, counts = np.unique(y, return_counts=True)
  complete_counts = []
  for c in classes:
    if c not in unique:
      complete_counts.append(0)
    else:
      index = np.where(unique == c)[0][0]
      complete_counts.append(counts[index])
  return np.array(complete_counts)



## split into train/test
def get_train_val_test_splits(X, y, max_points, seed, seed_batch,
                              split=(2./3, 1./6, 1./6)):
  """Return training, validation, and test splits for X and y.
  Args:
    X: features
    y: targets
    max_points: # of points to use when creating splits.
    seed: seed for shuffling.
    confusion: labeling noise to introduce.  0.1 means randomize 10% of labels.
    seed_batch: # of initial datapoints to ensure sufficient class membership.
    split: percent splits for train, val, and test.
  Returns:
    indices: shuffled indices to recreate splits given original input data X.
    y_copy: y, needed to reproduce results outside of
      run_experiments using original data.
  """
  np.random.seed(seed)
  X_copy = copy.copy(X)
  y_copy = copy.copy(y)

  indices = np.arange(len(y))

  if max_points is None:
    max_points = len(y_copy)
  else:
    max_points = min(len(y_copy), max_points)
  train_split = int(max_points * split[0])
  val_split = train_split + int(max_points * split[1])
  assert seed_batch <= train_split

  # Do this to make sure that the initial batch has examples from all classes
  min_shuffle = 3
  n_shuffle = 0
  y_tmp = y_copy

  # Need at least 4 obs of each class for 2 fold CV to work in grid search step
  while (any(get_class_counts(y_tmp, y_tmp[0:seed_batch]) < 4)
         or n_shuffle < min_shuffle):
    np.random.shuffle(indices)
    y_tmp = y_copy[indices]
    n_shuffle += 1

  X_train = X_copy[indices[0:train_split]]
  X_val = X_copy[indices[train_split:val_split]]
  X_test = X_copy[indices[val_split:max_points]]
  y_train = y_copy[indices[0:train_split]]
  y_val = y_copy[indices[train_split:val_split]]
  y_test = y_copy[indices[val_split:max_points]]
  # Make sure that we have enough observations of each class for 2-fold cv
  assert all(get_class_counts(y_copy, y_train[0:seed_batch]) >= 4)
  # Make sure that returned shuffled indices are correct
  assert all(y_copy[indices[0:max_points]] ==
             np.concatenate((y_train, y_val, y_test), axis=0))
  return (indices[0:max_points], X_train, y_train,
          X_val, y_val, X_test, y_test)



def select_batch(sampler, mixture, N, already_selected,
                   **kwargs):

    kwargs["N"] = 1
    kwargs["already_selected"] = already_selected
    batch_AL = sampler.select_batch(**kwargs)

    return batch_AL

seed = 1
np.random.seed(seed)


#--------------------------------------------------------------------------------------------------
subject = [101, 102, 103, 104, 105, 106, 107, 108, 109]
data_splits = [1., 0., 0.]


results_per_participant = csv.writer(open("./results/PAMAP2_RF100_results_by_participant_OAL.csv", "w"))
results_per_participant.writerow(["Participant", "Precision", "Recall", "F-Score", "Train Data Size"])

count = 0

gamma = 8
for p in subject:
    
    warmstart_size = 0.02
    batch_size = 0.02

    train_data = np.empty((0,86))
    print ("-----------------------------------------" )
    print ("Participant: " + str(p))
    test_data = pd.read_csv("./datasets/PAMAP2_Dataset/Features/PAMAP2_subject" + str(p) + ".features.csv", header = None, index_col = None)

    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1] 
    
    tr = np.delete(subject,count)
    print("Training on: " + str(tr))
    for t in tr:
        data = pd.read_csv("./datasets/PAMAP2_Dataset/Features/PAMAP2_subject" + str(t) + ".features.csv", header = None, index_col = None)
        train_data = np.vstack((train_data, data))
 
#   
    np.nan_to_num(train_data, copy = False)
    np.nan_to_num(test_data, copy = False)

    X_train = train_data[:,:-1]
    y_train = train_data[:,-1]


    max_points = len(y_train)
    train_size = int(min(max_points, len(y_train)))
    batch_size = int(batch_size * train_size)    
    seed_batch = int(warmstart_size * train_size)
    seed_batch = max(seed_batch, 6 * len(np.unique(y_train)))

    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1] 

    results = {}
    data_sizes = []
    accuracy = []
    selected_inds = list(range(seed_batch))
      
    select_model = score_model = RandomForestClassifier(n_estimators=100)
    
    sampler =  OALSampler(X_train, y_train, seed, gamma)
   
    n_samples = train_size - seed_batch+1
    print ("Entering batch for loop")
    requested = True     
    for b in range(n_samples):
        print ("Sample " + str(b))

        if (requested):
            requested = False
            n_train = len(selected_inds)
            print("Training model on " + str(n_train) + "/" + str(train_size) + " datapoints")

            # Sort active_ind so that the end results matches that of uniform sampling
            partial_X = X_train[sorted(selected_inds)]
            partial_y = y_train[sorted(selected_inds)]
            score_model.fit(partial_X, partial_y)
            select_model.fit(partial_X, partial_y)


            #           Test the model
            
            pred = select_model.predict(X_test)

            report = classification_report(y_test, pred)
            #print(report)
            dataframe = classification_report_csv(p, report)
          
             
            precision = dataframe.loc[len(dataframe)-1,'precision']
            recall = dataframe.loc[len(dataframe)-1,'recall']
            fscore = dataframe.loc[len(dataframe)-1 ,'f1_score']
            
            results_per_participant.writerow([str(p), str(precision), str(recall), str(fscore), str(float(n_train)/float(train_size)*100)])

        #print(train_size)
        n_sample = b*1
        select_batch_inputs = {
                "model": select_model,
                "sample": X_train[seed_batch+n_sample-1].reshape(1,-1),
                "labeled": dict(zip(selected_inds, y_train[selected_inds])),
                "y": y_train
            }
            
        if (select_batch(sampler, 1.0, n_sample,
                                     selected_inds, **select_batch_inputs)):
            requested = True
            new_sample = seed_batch + n_sample
            selected_inds.extend([new_sample])

            print('Requested: %d, Selected: %d' % (1, len([new_sample] )))
#

        #print (len(list(set(selected_inds))), len(selected_inds))
        assert len(list(set(selected_inds))) == len(selected_inds)
    
    count = count + 1

    

