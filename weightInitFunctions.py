import numpy as np

def zero_init(number_of_features_per_sample, number_of_classes):
    return np.zeros((number_of_features_per_sample, number_of_classes) )

def uniform_init(number_of_features_per_sample, number_of_classes, range=0.01):
    return np.random.uniform(low=-range, high=range, size=(number_of_features_per_sample, number_of_classes))

def normal_init(number_of_features_per_sample, number_of_classes, mean=0, std=1):
    return np.random.normal(loc=mean, scale=std, size=(number_of_features_per_sample, number_of_classes))
