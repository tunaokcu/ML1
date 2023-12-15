from Tester import Tester
import numpy as np
from saveLoad import ModelSaveLoad 
import pickle
from LOG import MultinomialLog
import matplotlib.pyplot as plt

from weightInitFunctions import zero_init, normal_init, uniform_init

class Optimizer:
    def __init__(self, initialized_model, configurations=[]):
        self.model = initialized_model
        self.configurations = configurations

    def optimize(self, X_test, y_test):
        self.accuracies = []

        self.best_accuracy = 0
        self.best_matrix = None
        self.best_weights = None
        self.best_configuration = -1

        for (i, configuration) in enumerate(self.configurations):
            title =  self.__encode(configuration)

            #Train and test model
            self.model.train(*configuration)
            tester = Tester(X_test, y_test, self.model)
            tester.test()
            self.accuracies.append(tester.accuracy)

            #Check accuracy, if better than the rest, set this as the current best_accuacy, best_matrix, best_weights
            if tester.accuracy > self.best_accuracy:
                self.best_accuracy = tester.accuracy
                self.best_matrix = np.copy(tester.confusion_matrix)
                self.best_weights = np.copy(self.model.weights)
                self.best_configuration = i

            #Save
            ModelSaveLoad.save(self.model, self.__encode(configuration))

 
        self.result = {"best_accuracy":self.best_accuracy, "best_matrix":self.best_matrix, "best_configuration":self.configurations[self.best_configuration], "all_configurations":self.configurations, "all_accuracies":self.accuracies}
        pickle.dump(self.result, open("OPTIMAL", 'wb'))

    #def load_optimal(self):
    def customized(self, X_test, y_test, load=False):
        if load:
            #        pickle.dump({"accuracies":accuracies, "titles": titles, "global_best_hyperparameters": global_best_hyperparameters}, open("OPTIMAL", 'wb'))
            pickled = pickle.load(open("OPTIMAL", 'rb'))
            #fixing titles
            titles = pickled["titles"]
            titles[1] = "Batch size 1"
            titles[2] = "Batch size 64"
            titles[3] =  "Batch size 50000"

            global_best_hyperparameters = pickled["global_best_hyperparameters"]
            accuracies = pickled["accuracies"]
            
            fig, axes = plt.subplots(7, 2, figsize=(12, 18))  # 7 rows, 2 columns
            axes = axes.flatten()
            for i in range(len(titles)):
                axes[i].plot(accuracies[i])
                axes[i].set_title(titles[i])
            plt.tight_layout()
            plt.show()

            #Q 2.3
            self.model.train(*global_best_hyperparameters)
            tester = Tester(X_test, y_test, self.model)
            tester.test()
            print("Optimal accuracy is", tester.accuracy)
            print("Confusion matrix is", tester.confusion_matrix)

            #Q 2.4
            self.visualize_weights(self.model.weights)

            #Q 2.5
            print(tester.scores)


        #set default 
        #def train(self, weight_init_function, epoch, batch_size, learning_rate, regularization_coefficient, keep_accuracies=False, X_test=None, y_test=None):
    
        default_configuration =  [normal_init, 100, 200, 5*(10**-4), (10**-4), True, X_test, y_test]
        accuracies = [self.model.train(*default_configuration)]
        titles = ["Default"]

        tester = Tester(X_test, y_test, self.model)

        #Q 2.1
        print("Default accuracy is", accuracies[-1][-1])
        print("Confusion matrix is", tester.calculate_and_return_confusion_matrix())

        global_best_accuracy = -1
        global_best_hyperparameters = None


        best_accuracy = -1
        best_batch_size = -1
        #first, pick best batch_size: 1, 64, 50000
        for batch_size in [1, 64, 50000]:
            default_configuration[2] = batch_size
            accuracies.append(self.model.train(*default_configuration))
            titles.append(f"Batch Size = {batch_size}")
            if accuracies[-1][-1] > best_accuracy:
                best_accuracy = accuracies[-1][-1]
                best_batch_size = batch_size

            if accuracies[-1][-1] > global_best_accuracy:
                global_best_accuracy = accuracies[-1][-1]
                global_best_hyperparameters = default_configuration[:]

        #set batch size to best_batch size
        default_configuration[2] = best_batch_size

        best_accuracy = -1
        best_technique = None
        #best technique
        for technique in [zero_init, uniform_init, normal_init]:
            default_configuration[0] = technique
            accuracies.append(self.model.train(*default_configuration))
            titles.append(f"Batch Size={default_configuration[2]} Technique={technique.__qualname__}")

            if accuracies[-1][-1] > best_accuracy:
                best_accuracy = accuracies[-1][-1]
                best_technique = technique


            if accuracies[-1][-1] > global_best_accuracy:
                global_best_accuracy = accuracies[-1][-1]
                global_best_hyperparameters = default_configuration[:]


        default_configuration[0] = best_technique

        best_accuracy = -1
        best_r = -1
        #best learning rate
        for learning_r in [0.1, 10**-3, 10**-4, 10**-5]:
            default_configuration[3] = learning_r
            accuracies.append(self.model.train(*default_configuration))
            titles.append(f"Batch Size={default_configuration[2]} Technique={default_configuration[0].__qualname__} Learning Rate={learning_r}")

            if accuracies[-1][-1] > best_accuracy:
                best_accuracy = accuracies[-1][-1]
                best_r = learning_r  

            if accuracies[-1][-1] > global_best_accuracy:
                global_best_accuracy = accuracies[-1][-1]
                global_best_hyperparameters = default_configuration[:]

        default_configuration[3] = best_r

        best_accuracy = -1
        best_coeff = -1
        #best learning rate
        for coeff in [10**-2, 10**-4, 10**-9]:
            default_configuration[4] = coeff
            accuracies.append(self.model.train(*default_configuration))
            titles.append(f"Batch Size={default_configuration[2]} Technique={default_configuration[0].__qualname__} Learning Rate={default_configuration[3]} Reg Coeff = {coeff}")

            if accuracies[-1][-1] > best_accuracy:
                best_accuracy = accuracies[-1][-1]
                best_coeff = coeff  

            if accuracies[-1][-1] > global_best_accuracy:
                global_best_accuracy = accuracies[-1][-1]
                global_best_hyperparameters = default_configuration[:]

        default_configuration[4] = best_coeff

        pickle.dump({"accuracies":accuracies, "titles": titles, "global_best_hyperparameters": global_best_hyperparameters}, open("OPTIMAL", 'wb'))

        #Q 2.2
        for i, accuracy in enumerate(accuracies):
            plt.figure(figsize=(10, 10))

            plt.subplot(4, 4, i+1)#we need 14 total
            plt.plot(accuracy)
            plt.title(titles[i])
        plt.subplots_adjust(wspace=3) 
        plt.show()

        #Q 2.3
        self.model.train(*global_best_hyperparameters)
        tester = Tester(X_test, y_test, self.model)
        tester.test()
        print("Optimal accuracy is", global_best_accuracy)
        print("Confusion matrix is", tester.confusion_matrix)

        #Q 2.4
        self.visualize_weights(self.model.weights)

        #Q 2.5
        print(tester.scores)



    def visualize_weights(self, weight):
        plt.matshow(weight, cmap=plt.cm.gray, vmin=0.5*weight.min(), vmax=0.5*weight.max())

    def __encode(self, configuration):
        return ','.join(x.__qualname__ if hasattr(x, '__call__') else str(x)  for x in configuration) + ".npy"
    
    def __decode(self, title):
        without_npy = title[:-4]
        return [float(s) for s in without_npy.split(',')]