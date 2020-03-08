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
