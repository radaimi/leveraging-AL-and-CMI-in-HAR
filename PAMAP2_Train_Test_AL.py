#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 18:42:21 2018

@author: Rebecca Adaimi

PAMAP2 Training and Testing AL

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
from informative_diverse import InformativeClusterDiverseSampler
from uniform_sampling import UniformSampling
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



def select_batch(sampler, N, already_selected,
                   **kwargs):
    n_active = int(N)
    kwargs["N"] = n_active
    kwargs["already_selected"] = already_selected
    batch_AL,m = sampler.select_batch(**kwargs)
    already_selected = already_selected + batch_AL
    return batch_AL

seed = 1
np.random.seed(seed)


#--------------------------------------------------------------------------------------------------
subject = [101, 102, 103, 104, 105, 106, 107, 108, 109]
data_splits = [1., 0., 0.]

file = open("./results/PAMAP2_RF100_results_AL.csv", "w")
results_per_participant = csv.writer(file)
results_per_participant.writerow(["Participant", "Precision", "Recall", "F-Score", "Train Data Size"])

count = 0

dict_AV_FSCORE = {}
dict_AV_RECALL = {}
dict_AV_PRECISION = {}
for p in subject:
    
    file1 = open("./results/PAMAP2_results_test_participant_" + str(p) + "_indxs_selected_AL.csv", "w")
    indxs = csv.writer(file1)
    indxs.writerow(["Indices Selected", "Train Data Size"])

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

    train_size = int(len(y_train))
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
    
    sampler =  InformativeClusterDiverseSampler(X_train, y_train, seed)
        
            
    n_batches = int(np.ceil((1.0 * train_size - seed_batch) * 1.0 / batch_size)) + 1
    for b in range(n_batches):
        n_train = seed_batch + min(train_size - seed_batch, b * batch_size)
        print("Training model on " + str(n_train) + "/" + str(train_size) + " datapoints")
    
        assert n_train == len(selected_inds)
        data_sizes.append(n_train)
        
        # Sort active_ind so that the end results matches that of uniform sampling
        partial_X = X_train[sorted(selected_inds)]
        partial_y = y_train[sorted(selected_inds)]
        score_model.fit(partial_X, partial_y)
        select_model.fit(partial_X, partial_y)
        
        
        pred = select_model.predict(X_test)

            
        report = classification_report(y_test, pred)
        #print(report)
        dataframe = classification_report_csv(p, report)         
        precision = dataframe.loc[len(dataframe)-1,'precision']
        recall = dataframe.loc[len(dataframe)-1,'recall']
        fscore = dataframe.loc[len(dataframe)-1 ,'f1_score']
        print("Precision: {}, Recall: {}, F-Score: {}".format(precision, recall, fscore))
        results_per_participant.writerow([str(p), str(precision), str(recall), str(fscore), str(float(n_train)/float(train_size)*100)])

        #print(train_size)
        #print(batch_size, train_size - len(selected_inds))
        n_sample = min(batch_size, train_size - len(selected_inds))
        #print(n_sample)
        select_batch_inputs = {
                "model": select_model,
                "labeled": dict(zip(selected_inds, y_train[selected_inds])),
                "y": y_train
            }
        new_batch = select_batch(sampler, n_sample,
                                     selected_inds, **select_batch_inputs)
        
        if (b == 0):
            indxs.writerow([str(selected_inds), str(n_train)])
        else:
            indxs.writerow([str(new_batch), str(n_train)])
        selected_inds.extend(new_batch)
        print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
        assert len(new_batch) == n_sample
        #print (len(list(set(selected_inds))), len(selected_inds))
        assert len(list(set(selected_inds))) == len(selected_inds)
        dict_AV_FSCORE[b] = dict_AV_FSCORE.get(b,0) + fscore
        dict_AV_RECALL[b] = dict_AV_RECALL.get(b,0) + recall
        dict_AV_PRECISION[b] = dict_AV_PRECISION.get(b,0) + precision

    file1.close()
      
    count = count + 1
file.close()

dicts = dict_AV_FSCORE, dict_AV_RECALL, dict_AV_PRECISION
res_array = np.empty((0,3))
with open('./results/PAMAP2_results_AL_Average_result.csv', 'w') as ofile:
    writer = csv.writer(ofile)
    writer.writerow(['Iteration', 'F-Score', 'Recall', 'Precision'])
    for key in dict_AV_FSCORE.keys():
        row = [d[key]/len(subject) for d in dicts]
        res_array = np.vstack((res_array, row))
        row = [key] + row
        writer.writerow(row)
    
    
plt.figure()
x = np.append(np.arange(1.99,101.,2.),100.)
plt.plot(x,res_array[:,0])
plt.plot(x,res_array[:,1])
plt.plot(x,res_array[:,2])
plt.legend(["Average F-score", "Average Recall", "Average Precision"])
plt.title("PAMAP2")
plt.xlabel('Percentage of Training Data')
plt.show()

