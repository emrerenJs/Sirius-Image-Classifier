from enum import Enum

class DataInformations(Enum):
    #Classifiers
    CLASSIFIER_ANN = 0
    CLASSIFIER_RF = 1
    #Datasets
    DATASET_ORIGINAL = 0
    DATASET_AUGMENTATED_AND_ORIGINAL = 1
    #Deep learning algorithms
    DL_VGG16 = 0
    DL_RESNET50 = 1