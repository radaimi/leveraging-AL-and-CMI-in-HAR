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
    user_data_file = '.\ExtraSensory.per_uuid_features_labels\%s.features_labels.csv.gz' % uuid;

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
    for (idx, sensor) in enumerate(sensors_to_use.split(' ')):
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

def train_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label):
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
    
    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.
    
    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    lr_model = sklearn.linear_model.LogisticRegression(class_weight='balanced');
    lr_model.fit(X_train,y);
    
    # Assemble all the parts of the model:
    model = {\
            'sensors_to_use':sensors_to_use,\
            'target_label':target_label,\
            'mean_vec':mean_vec,\
            'std_vec':std_vec,\
            'lr_model':lr_model};
    
    return model;

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

def test_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model,fold):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    print('X_test' , np.shape(X_test))
    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],model['sensors_to_use']));
    print('X_test' , np.shape(X_test))
    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec']);
    print('X_test' , np.shape(X_test))
    # The single target label:
    label_ind = label_names.index(model['target_label']);
    print(np.shape(Y_test))
    y = Y_test[:,label_ind];
    print(np.shape(y))
    missing_label = M_test[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label,:];
    print('X_test' , np.shape(X_test))
    y = y[existing_label];
    print(np.shape(y))
    timestamps = timestamps[existing_label];

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.;
    print('X_test' , np.shape(X_test))
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
                                      str(balanced_accuracy), str(precision), str(fscore)])

    
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
path_cv5 = ".\cv5Folds\cv_5_folds"

cv5_files = os.listdir(path_cv5)
count = 0



results_per_participant = csv.writer(open("./results/results_PL_FIXED.csv", "w"))
results_per_participant.writerow(["Fold", "Sensor Used", "Target Label", "Accuracy", "Sensitivity (TPR)", 
                                  "Specificity (TNR)", "Balanced Accuracy", "Precision", "F1-Score"])

    
X_train = np.empty((0,225))
y_train = np.empty((0,51))
X_test = np.empty((0,225))
y_test = np.empty((0,51))
M_train = np.empty((0,51))
M_test = np.empty((0,51))
timestamps_test = []

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

    if count % 4 == 0 and modeled == False:
        print("-----------------------------------------------------------------------------------------")
        print ("Fold " + str(fold) + ": Model Training and Testing")
        for (idx, sensors_to_use) in enumerate(sensors):
            for (idx, target_labels) in enumerate(context_labels):
                feat_sensor_names = get_sensor_names_from_features(feature_names);
                model = train_model(X_train,y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_labels)
                test_model(X_test,y_test,M_test,timestamps_test,feat_sensor_names,label_names,model,fold);
                
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
               
        else:
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
for (idx, sensors_to_use) in enumerate(sensors):
    for (idx, target_labels) in enumerate(context_labels):
        feat_sensor_names = get_sensor_names_from_features(feature_names);
        model = train_model(X_train,y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_labels)
        test_model(X_test,y_test,M_test,timestamps_test,feat_sensor_names,label_names,model,fold);





