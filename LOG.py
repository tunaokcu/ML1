import matplotlib.pyplot as plt
import numpy as np
import gzip
import math
from tqdm import tqdm 
from weightInitFunctions import zero_init, normal_init, uniform_init
from saveLoad import ModelSaveLoad 
from Tester import Tester
import Optimizer

# One-hot encoding of the labels
def one_hot_encoding(label_data):
    return np.eye(10)[label_data]

# Function to read pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    return normalized_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    return label_data

    one_hot_encoding_labels = one_hot_encoding(label_data)
    return one_hot_encoding_labels

# Function to read the entire dataset
def read_dataset():
    X_train = read_pixels("data/train-images-idx3-ubyte.gz")
    y_train = one_hot_encoding(read_labels("data/train-labels-idx1-ubyte.gz"))
    X_test = read_pixels("data/t10k-images-idx3-ubyte.gz")
    y_test = read_labels("data/t10k-labels-idx1-ubyte.gz")

    number_of_images = y_train.shape[0]
    number_of_pixels = X_train.shape[0]
    number_of_pixels_per_image = number_of_pixels // number_of_images
    
    #reshape X_train and X_test
    X_train = X_train.reshape((number_of_images, number_of_pixels_per_image))
    X_test = X_test.reshape(y_test.shape[0], number_of_pixels_per_image)

    #!add a column of 1's to the features to include bias
    X_train = np.c_[X_train, np.ones(X_train.shape[0])]
    X_test = np.c_[X_test, np.ones(X_test.shape[0])]

    return X_train, y_train, X_test, y_test



class MultinomialLog:
    def __init__(self, X_train, y_train, number_of_classes):
        self.trained = False
        self.X_train = X_train#.astype(np.float128)
        self.y_train = y_train#.astype(np.float128)

        self.number_of_samples, self.number_of_features_per_sample = X_train.shape
        self.number_of_classes = number_of_classes


    def train(self, weight_init_function, epoch, batch_size, learning_rate, regularization_coefficient, keep_accuracies=False, X_test=None, y_test=None):
        self.weights = weight_init_function(self.number_of_features_per_sample, self.number_of_classes)#.astype(np.float128)
        #!our current approach can result in batch sizes less than batch_size
        #?do this or wrap around or something?            
        
        tester = Tester(X_test, y_test, self)
        accuracies = [tester.calculate_and_return_accuracy()]

        
        for _ in tqdm(range(epoch)):
            for start in range(math.ceil(self.number_of_samples / batch_size)):
                end = min(start + batch_size, self.number_of_samples) #for when batch_size causes this batch to overflow
                #now all we need to do is update our weights and biases on X[start:end] and Y[start:end]
                #this seems about right https://math.stackexchange.com/questions/1652661/gradient-descent-l2-norm-regularization
                self.weights -= learning_rate*self.__gradient2(self.X_train[start:end,:], self.y_train[start:end,:], regularization_coefficient)#.__gradient(self.X_train[start:end,:], self.__activation(self.X_train[start:end,:]), self.y_train[start:end,:], regularization_coefficient)

            if (keep_accuracies):
                accuracies.append(tester.calculate_and_return_accuracy())

        self.trained = True

        return accuracies

    def predict(self, X):
        return np.argmax(self.__activation(X), axis=1)
    
    def __gradient2(self, X_train, y_train, regularization_coefficient ):
        error= self.__activation(X_train) - y_train
        regularization_term = np.vstack((self.weights[:-1], np.zeros((1, self.weights.shape[1]))))
        #np.vstack((np.zeros((1, weights.shape[1])), weights[1:]))
        rest = np.dot(X_train.T, error)
        return rest + regularization_term*regularization_coefficient
    
    def __gradient(self, X_train, probabilities, y_train, regularization_coefficient):
        error = probabilities - y_train
        without_regularization =  np.dot(X_train.T, error) / self.number_of_samples

        regularization_term = (regularization_coefficient) * np.concatenate((np.zeros((1, 10)), self.weights[:-1]))  # Exclude the bias 
        return without_regularization + regularization_term


    def __activation(self, X):
        return softmax(np.dot(X, self.weights))
    

def softmax(X):
    exp_z = np.exp(X - np.max(X, axis=1, keepdims=True))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)



def main():
    X_train, y_train, X_test, y_test = read_dataset()
    
    default_model = MultinomialLog(X_train, y_train, 10)
    optimizer = Optimizer.Optimizer(default_model)
    optimizer.customized(X_test, y_test, True)

    """
    default_model = MultinomialLog(X_train, y_train, 10)
    default_optimizer = Optimizer(default_model)
    default_configuration =  [normal_init, 100, 200, 5*(10**-4), (10**-4)]
    default_optimizer.configurations.append(default_configuration)
    default_optimizer.optimize(X_test, y_test)

    #def train(weight_init_function, epoch, batch_size, learning_rate, regularization_coefficient):
    #optimizer.configurations = [[normal_init, 100, 50000, 10**-2, 10**-2], [normal_init, 100, 50000, 10**-3, 10**-2], [normal_init, 100, 50000, 10**-4, 10**-2]]
    model = MultinomialLog(X_train, y_train, 10)
    optimizer = Optimizer(model)
    default
    """
if __name__ == "__main__":
    main()

