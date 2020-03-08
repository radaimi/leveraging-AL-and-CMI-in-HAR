#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 18:42:21 2018

@author: Rebecca Adaimi

PAMAP2 Training and Testing

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
import random
from sklearn.metrics import classification_report, confusion_matrix
import sklearn


def classification_report_csv(subject, report):
    print(report)
    report_data = []
    lines = report.split('\n')
#    print(lines)
#    print(lines)
    for line in lines[2:-5]:
#        print(line)
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
        #print(line)
        row = {}
        row_data = line.split('      ')
        #print(row_data)
        row['Participant'] = np.nan
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)

    dataframe = pd.DataFrame.from_dict(report_data)
    #print(dataframe)
    return dataframe

seed = 1
np.random.seed(seed)


#--------------------------------------------------------------------------------------------------
subject = [101, 102, 103, 104, 105, 106, 107, 108, 109]
count = 0

print (sklearn.__version__)
report_df = pd.DataFrame()

AV_PRECISION = 0
AV_RECALL = 0
AV_FSCORE = 0
for p in subject:

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
    
    np.nan_to_num(train_data, copy = False)
    np.nan_to_num(test_data, copy = False)
    X_train = train_data[:,:-1]
    y_train = train_data[:,-1]
    
    model = RandomForestClassifier(100)
    model.fit(X_train, y_train)

    
    pred = model.predict(X_test)
    #print(Counter(y_test))
    report = classification_report(y_test, pred)
    #print(report)
    dataframe = classification_report_csv(p, report)
    report_df = report_df.append(dataframe)
    
    # Show confusion matrix in a separate window
    # plt.matshow(cm)
    # plt.title('Confusion matrix')
    # plt.colorbar()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()
 
    AV_PRECISION += dataframe['precision'].iloc[-1]
    AV_RECALL += dataframe['recall'].iloc[-1]
    AV_FSCORE += dataframe['f1_score'].iloc[-1]

    count = count + 1

print('AVERAGE FSCORE: {}'.format(AV_FSCORE/count))
print('AVERAGE PRECISION: {}'.format(AV_PRECISION/count))
print('AV_RECALL: {}'.format(AV_RECALL/count))

#print(report_df)
report_df.to_csv('./Results/PAMAP2_RF100_classification_report_PL.csv', index = False)




