# Leveraging Active Learning and Conditional Mutual Information to Minimize Data Annotation in Human Activity Recognition
This repo contains code accompanying the paper, "Leveraging Active Learning and Conditional Mutual Information to Minimize Data Annotation in Human Activity Recognition". 

## Datasets 

- [PAMAP2 dataset](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)
- [Opportuniy dataset](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition)
- [ExtraSensory dataset](http://extrasensory.ucsd.edu) -- specifically the cross validation partition.
- Fluid Intake dataset

## Scripts

### Sampling Scripts
The informative and diverse pool-based sampling is based on https://github.com/google/active-learning with modifications that made it faster with large datasets (ExtraSensory) and made it work with matlab scripts (Opportunity)
- [informative_diverse.py](informative_diverse.py): implements pool-based AL sampling 
- [OAL_sampling.py](OAL_sampling.py): implements stream-based AL (online AL)

### PAMAP2-related Scripts
- [PAMAP2_processing.py](PAAMP2_processing.py): loads the data, preprocesses the data, and extracts features. The features are saved in csv files.
- [PAMAP2_train_test_PL.py](PAMAP2_train_test_PL.py): implements the fully supervised training and testing of the data. Check paper for details.
- [PAMAP2_train_test_AL.py](PAMAP2_train_test_AL.py): implements the iterative pool-based AL.  
- [PAMAP2_train_test_OAL.py](PAMAP2_train_test_OAL.py): implements the iterative stream-based AL.  

### ExtraSensory-related Scripts
- [PL_single_sensors_main.py](PL_single_sensors_main.py): implements the fully supervised training and testing of the data using the single-sensor approach. 
- [PL_EF_classification_main.py](PL_EF_classification_main.py): implements the fully supervised training and testing of the data using the early-fusion (EF) approach.
- [AL_single_sensors_main.py](AL_single_sensors_main.py): implements the iterative pool-based AL for the single-sensor approach.  
- [AL_EF_classification_main.py](AL_EF_classification_main.py): implements the iterative pool-based AL for the early-fusion (EF) approach. 
- [OAL_single_sensors.py](OAL_single_sensors.py): implements the iterative stream-based AL for the single-sensor approach. 
- [OAL_EF.py](OAL_EF.py): implements the iterative stream-based AL for the early-fusion (EF) approach. 


## Reference

Rebecca Adaimi and Edison Thomaz. 2019. Leveraging Active Learning and Conditional Mutual Information to Minimize Data Annotation in Human Activity Recognition. <i>Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.</i> 3, 3, Article 70 (September 2019), 23 pages.

[Download paper here](https://sites.google.com/view/rebecca-adaimi/active-learning?authuser=0)

Bibtex Reference:
```
@article{10.1145/3351228,
author = {Adaimi, Rebecca and Thomaz, Edison},
title = {Leveraging Active Learning and Conditional Mutual Information to Minimize Data Annotation in Human Activity Recognition},
year = {2019},
issue_date = {September 2019},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {3},
number = {3},
url = {https://doi.org/10.1145/3351228},
doi = {10.1145/3351228},
abstract = {A difficulty in human activity recognition (HAR) with wearable sensors is the acquisition of large amounts of annotated data for training models using supervised learning approaches. While collecting raw sensor data has been made easier with advances in mobile sensing and computing, the process of data annotation remains a time-consuming and onerous process. This paper explores active learning as a way to minimize the labor-intensive task of labeling data. We train models with active learning in both offline and online settings with data from 4 publicly available activity recognition datasets and show that it performs comparably to or better than supervised methods while using around 10% of the training data. Moreover, we introduce a method based on conditional mutual information for determining when to stop the active learning process while maximizing recognition performance. This is an important issue that arises in practice when applying active learning to unlabeled datasets.},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = sep,
articleno = {70},
numpages = {23},
keywords = {Stopping Criterion, Conditional Mutual Information, Active Learning, Human Activity Recognition, Data Annotation}
}
```