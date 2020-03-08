# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 19:18:54 2018

@author: Rebecca Adaimi
"""

import matplotlib.pyplot as plt;
import numpy as np
import gzip
from  io import StringIO, BytesIO
import os
import sklearn.linear_model
import csv

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from collections import Counter
from informative_diverse import InformativeClusterDiverseSampler
import random
import io
import copy
from collections import Counter

def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.decode('utf8').index('\n')];
    columns = headline.decode('utf8').split(',');

    # The first column should be timestamp:
    assert columns[0] == 'timestamp';
    # The last column should be label_source:
    assert columns[-1] == 'label_source';
    
    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci;
            break;
        pass;

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind];
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1];
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:');
        label_names[li] = label.replace('label:','');
        pass;
    
    return (feature_names,label_names);

def parse_body_of_csv(csv_str,n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(BytesIO(csv_str),delimiter=',',skiprows=1);
    
    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int);
    
    # Read the sensor features:
    X = full_table[:,1:(n_features+1)];
    
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1]; # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat); # M is the missing label matrix
    Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix
    
    return (X,Y,M,timestamps);

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''

def read_user_data(uuid):
    user_data_file = './ExtraSensory.per_uuid_features_labels/%s.features_labels.csv.gz' % uuid;

    # Read the entire csv file of the user:
    with gzip.open(user_data_file,'rb') as fid:
        csv_str = fid.read();
#        print(type(csv_str))
        pass;

    (feature_names,label_names) = parse_header_of_csv(csv_str);
    n_features = len(feature_names);
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features);

    return (X,Y,M,timestamps,feature_names,label_names);


#%%
 
def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names),dtype=bool);
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor);
        use_feature = np.logical_or(use_feature,is_from_sensor);
        pass;
    X = X[:,use_feature];
    return X;

def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train,axis=0);
    std_vec = np.nanstd(X_train,axis=0);
    return (mean_vec,std_vec);

def standardize_features(X,mean_vec,std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1,-1));
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1));
    X_standard = X_centralized / normalizers;
    return X_standard;


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

def get_train_val_test_splits(X, y, max_points, seed, confusion, seed_batch,
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
    y_noise: y with noise injected, needed to reproduce results outside of
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
  return X_train, y_train
  
def train_test_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label,seed, X_test,y_test,M_test,timestamps_test,fold):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train,feat_sensor_names,sensors_to_use);
    print("== Projected the features to %d features from the sensors: %s" % (X_train.shape[1], sensors_to_use));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec,std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train,mean_vec,std_vec);
    # The single target label:
    label_ind = label_names.index(target_label);
    y = Y_train[:,label_ind];
    missing_label = M_train[:,label_ind];
    existing_label = np.logical_not(missing_label);
    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label,:];
    y = y[existing_label];
#    y = y.reshape((-1,1))

    
    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    X_train[np.isnan(X_train)] = 0.;

    sampler =  InformativeClusterDiverseSampler(X_train, y, seed)

            
    warmstart_size = 0.02
    batch_size = 0.02
    
    
    max_points = len(X_train)
    train_size = int(min(max_points, len(X_train)))
    batch_size = int(batch_size * train_size)    
    seed_batch = int(warmstart_size * train_size)
    seed_batch = max(seed_batch, 6 * len(np.unique(y_train)))
    

    selected_inds = list(range(seed_batch))

    n_batches = int(np.ceil((1.0 * train_size - seed_batch) * 1.0 / batch_size)) + 1
    print ("Entering batch for loop")
    
    #X_train, y = get_train_val_test_splits(X_train, y, max_points, seed, 0, seed_batch, split=(1, 0, 0))
    train_data = np.hstack((X_train,y.reshape(-1,1)))
    pd.DataFrame(train_data).to_csv("./results/train_data_fold_" + str(fold) + "_target_label_" + str(target_label) + "_AL.csv")

    file1 = open("./results/selected_indx_fold_" + str(fold) + "_EF_target_label_" + str(target_label) + "_AL.csv", "w")
    indxs = csv.writer(file1)
    indxs.writerow(["Indices Selected", "Train Data Size"])
    for b in range(n_batches):
        print ("Batch " + str(b))
        n_train = seed_batch + min(train_size - seed_batch, b * batch_size)
    
        assert n_train == len(selected_inds)
        
        # Sort active_ind so that the end results matches that of uniform sampling
        partial_X = X_train[sorted(selected_inds)]
        partial_y = y[sorted(selected_inds)]

        print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
              (len(partial_y),(target_label),sum(partial_y),sum(np.logical_not(partial_y))) );

    
    
        # Now, we have the input features and the ground truth for the output label.
        # We can train a logistic regression model.
        
        # Typically, the data is highly imbalanced, with many more negative examples;
        # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
        if (len(Counter(partial_y)) > 1):
            lr_model = sklearn.linear_model.LogisticRegression(class_weight='balanced');
        else:
            results_per_participant.writerow([str(fold), "--","--", str("1-Class"),str("1-Class"), str("1-Class"), str("1-Class"), str("1-Class"), str("1-Class"), str(n_train)])
            break
        
        lr_model.fit(partial_X,partial_y);
    
        # Assemble all the parts of the model:
        model = {\
                'sensors_to_use':sensors_to_use,\
                'target_label':target_label,\
                'mean_vec':mean_vec,\
                'std_vec':std_vec,\
                'lr_model':lr_model};
        
        test_model(X_test,y_test,M_test,timestamps_test,feat_sensor_names,label_names,model,fold, n_train);
        
        n_sample = min(batch_size, train_size - len(selected_inds))
        select_batch_inputs = {
                "model": model['lr_model'],
                "labeled": dict(zip(selected_inds, X_train[selected_inds])),
                "y": y
            }
        new_batch, min_margin = select_batch(sampler, n_sample,
                                     selected_inds, **select_batch_inputs)

        indxs.writerow([str(selected_inds), str(n_train)])

        m = np.hstack((np.asarray(list(range(len(X_train)))).reshape(-1,1), np.asarray(min_margin)))
        pd.DataFrame(m).to_csv("./results/min_margin_EF_fold_" + str(fold) + "_target_label_" + target_label + "_n_iter_" + str(b) + "_informative_diverse.csv")
        
        selected_inds.extend(new_batch)
        print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
        assert len(new_batch) == n_sample
        print(len(list(set(selected_inds))))
        print(len(selected_inds))
        assert len(list(set(selected_inds))) == len(selected_inds)



def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names]);
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc';
            pass;
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro';
            pass;
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet';
            pass;
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc';
            pass;
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass';
            pass;
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud';
            pass;
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP';
            pass;
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS';
            pass;
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF';
            pass;
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat);

        pass;

    return feat_sensor_names;


def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking';
    if label == 'FIX_running':
        return 'Running';
    if label == 'LOC_main_workplace':
        return 'At main workplace';
    if label == 'OR_indoors':
        return 'Indoors';
    if label == 'OR_outside':
        return 'Outside';
    if label == 'LOC_home':
        return 'At home';
    if label == 'FIX_restaurant':
        return 'At a restaurant';
    if label == 'OR_exercise':
        return 'Exercise';
    if label == 'LOC_beach':
        return 'At the beach';
    if label == 'OR_standing':
        return 'Standing';
    if label == 'WATCHING_TV':
        return 'Watching TV'
    
    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;
    
    label = label.replace('__',' (').replace('_',' ');
    label = label[0] + label[1:].lower();
    label = label.replace('i m','I\'m');
    return label;

def test_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model,fold, n_train):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
#    print(np.shape(X_test))
    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use']);
#    print(np.shape(X_test))

    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],model['sensors_to_use']));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec']);
#    print(np.shape(X_test))

    # The single target label:
    label_ind = label_names.index(model['target_label']);
#    print(label_ind)
    y = Y_test[:,label_ind];
    missing_label = M_test[:,label_ind];
#    print(Counter(missing_label))

    existing_label = np.logical_not(missing_label);
#    print(Counter(existing_label))

    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label,:];
    y = y[existing_label];
    # timestamps = timestamps[existing_label];
#    print(np.shape(X_test))

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.;
#    print(np.shape(X_test))

    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y),get_label_pretty_name(model['target_label']),sum(y),sum(np.logical_not(y))) );
    try:
        # Preform the prediction:
        y_pred = model['lr_model'].predict(X_test);
        
        # Naive accuracy (correct classification rate):
        accuracy = np.mean(y_pred == y);
        
        # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
        tp = np.sum(np.logical_and(y_pred,y));
        tn = np.sum(np.logical_and(np.logical_not(y_pred),np.logical_not(y)));
        fp = np.sum(np.logical_and(y_pred,np.logical_not(y)));
        fn = np.sum(np.logical_and(np.logical_not(y_pred),y));
        
        
        accuracy = float(tp+tn) / (tp+tn+fp+fn)
        fscore = (2*precision*sensitivity) / (precision+sensitivity)
        # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
        
        # Balanced accuracy is a more fair replacement for the naive accuracy:
        balanced_accuracy = (sensitivity + specificity) / 2.;
        
        # Precision:
        # Beware from this metric, since it may be too sensitive to rare labels.
        # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
        # and for each label the pos/neg ratio is different.
        # This can cause undesirable and misleading results when averaging precision across different labels.
        precision = float(tp) / (tp+fp);
        
        print("-"*10);
        print('Accuracy*:         %.2f' % accuracy);
        print('Sensitivity (TPR): %.2f' % sensitivity);
        print('Specificity (TNR): %.2f' % specificity);
        print('Balanced accuracy: %.2f' % balanced_accuracy);
        print('Precision**:       %.2f' % precision);
        print('F1 Score:          %.2f' % fscore);
        print("-"*10);
        
    except:
        accuracy= np.nan
        sensitivity = np.nan
        specificity = np.nan
        balanced_accuracy = np.nan
        precision = np.nan
        fscore = np.nan
    
    results_per_participant.writerow([str(fold), model['sensors_to_use'], model['target_label'], 
                                      str(accuracy), str(sensitivity), str(specificity), 
                                      str(balanced_accuracy), str(precision), str(fscore), str(n_train)])
    print("results written in csv file")
      
    return;
#%%
    
    
"""
Context Labels used:
    - Lying Down
    - Sitting
    - Walking 
    - Running
    - Bicycling
    - Sleeping
    - Lab work
    - In class
    - In a meeting
    - At main workplace
    - Indoors
    - Outside
    - In a car
    - Drive (I'm the driver)
    - Drive (I'm a passenger)
    - At home
    - At a restaurant
    - Phone in pocket
    - Exercise
    - Cooking
    - Strolling
    - Bathing - Shower
"""

# 5-fold Cross Validation files
path_cv5 = "./cv5Folds/cv_5_folds"

cv5_files = os.listdir(path_cv5)
cv5_files.sort()
count = 0
modeled = False
def select_batch(sampler, N, already_selected,
                   **kwargs):
    n_active = int(N)
    print(n_active)

    n_passive = N - n_active
    print(n_passive)
    kwargs["N"] = n_active
    kwargs["already_selected"] = already_selected
    batch_AL, min_margin = sampler.select_batch(**kwargs)
    already_selected = already_selected + batch_AL
    return batch_AL, min_margin


results_per_participant = csv.writer(open("./results/results_AL_Informative_Diverse_EF_classification.csv", "w"))
results_per_participant.writerow(["Fold", "Sensor Used", "Target Label", "Accuracy", "Sensitivity (TPR)", 
                                  "Specificity (TNR)", "Balanced Accuracy", "Precision", "F1-Score","Train Size"])

    
X_train = np.empty((0,225))
y_train = np.empty((0,51))
X_test = np.empty((0,225))
y_test = np.empty((0,51))
M_train = np.empty((0,51))
M_test = np.empty((0,51))
timestamps_test = []

seed = 1
np.random.seed(seed)
context_labels = ['LYING_DOWN','SITTING','FIX_walking','FIX_running','BICYCLING','SLEEPING','LAB_WORK',
                 'IN_CLASS','IN_A_MEETING','LOC_main_workplace','OR_indoors','OR_outside','IN_A_CAR',
                 ,'DRIVE_-_I_M_THE_DRIVER','DRIVE_-_I_M_A_PASSENGER','LOC_home','FIX_restaurant',
                 'PHONE_IN_POCKET','OR_exercise','COOKING','STROLLING',
                 'BATHING_-_SHOWER']


sensors = ['Acc','Gyro','WAcc','Loc','Aud','PS']
modeled = True
fold = 0
while count < len(cv5_files):

    cv = cv5_files[count]
    if count % 4 == 0  and modeled == False:

        print("-----------------------------------------------------------------------------------------")
        print ("Fold " + str(fold) + ": Model Training and Testing")
        sensors_to_use = sensors
        for (idx, target_labels) in enumerate(context_labels):
            feat_sensor_names = get_sensor_names_from_features(feature_names);
            train_test_model(X_train,y_train,M_train,feat_sensor_names,label_names,sensors_to_use,
                                 target_labels, seed, X_test,y_test,M_test,timestamps_test,fold)
                
        X_train = np.empty((0,225))
        y_train = np.empty((0,51))
        X_test = np.empty((0,225))
        y_test = np.empty((0,51))
        M_train = np.empty((0,51))
        M_test = np.empty((0,51))
        timestamps_test = []
        
        fold += 1
        print ("----------------------------------------------------------------------------------------")
        modeled = True
    else:
        modeled = False
        print(cv)
    
        cv_path = os.sep.join((path_cv5,cv))
    #    print(cv_path)
    
        if str(fold) + '_train' in cv:
                file = np.loadtxt(cv_path,dtype=np.str)  
                # load data for training 
                for uuid in file:
                    print('Training Data: ', uuid)
                    (X,y,M,timestamps,feature_names,label_names) = read_user_data(uuid)
                    X_train = np.vstack((X_train, X))
                    y_train = np.vstack((y_train, y))
                    M_train = np.vstack((M_train, M))
               
        elif str(fold) + '_test' in cv:
                file = np.loadtxt(cv_path,dtype=np.str)  
                # load data for testing
                for uuid in file:
                    print('Testing Data: ', uuid)
                    (X,y,M,timestamps,feature_names,label_names) = read_user_data(uuid)
                    X_test = np.vstack((X_test, X))
                    y_test = np.vstack((y_test, y))
                    M_test = np.vstack((M_test, M))
                    timestamps_test = np.append(timestamps_test, timestamps)

        count += 1

print("-----------------------------------------------------------------------------------------")
print ("Fold " + str(fold) + ": Model Training and Testing")
sensors_to_use = sensors
for (idx, target_labels) in enumerate(context_labels):
    feat_sensor_names = get_sensor_names_from_features(feature_names);
    train_test_model(X_train,y_train,M_train,feat_sensor_names,label_names,sensors_to_use,
                     target_labels, seed, X_test,y_test,M_test,timestamps_test,fold)



                





