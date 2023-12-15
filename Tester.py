import numpy as np
from tqdm import tqdm 


class Tester:
    def __init__(self, X_test, y_test, model):
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

    def calculate_and_return_accuracy(self):
        self.accuracy = 0
        y_pred = self.model.predict(self.X_test)
        for ground_truth, prediction in tqdm(zip(self.y_test, y_pred), total=y_pred.shape[0]):

            if ground_truth == prediction:
                self.accuracy += 1
        
        self.accuracy = self.accuracy / y_pred.shape[0]
        return self.accuracy
    
    def calculate_and_return_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        self.confusion_matrix = np.zeros((self.model.number_of_classes, self.model.number_of_classes), dtype=int)

        for ground_truth, prediction in tqdm(zip(self.y_test, y_pred), total=y_pred.shape[0]):
            self.confusion_matrix[ground_truth, prediction] += 1

        return self.confusion_matrix

    def test(self):
        y_pred = self.model.predict(self.X_test)
        self.confusion_matrix = np.zeros((self.model.number_of_classes, self.model.number_of_classes), dtype=int)
        self.labeled_confusion_matrix = {i:{"tp":0, "fp":0, "tn":0, "fn":0} for i in range(self.model.number_of_classes)}
        self.accuracy = 0

        for ground_truth, prediction in tqdm(zip(self.y_test, y_pred), total=y_pred.shape[0]):
            self.confusion_matrix[ground_truth, prediction] += 1

            if ground_truth == prediction:
                self.labeled_confusion_matrix[ground_truth]["tp"] += 1
                self.accuracy += 1
            elif ground_truth != prediction:
                self.labeled_confusion_matrix[ground_truth]["fn"] += 1
                self.labeled_confusion_matrix[prediction]["fp"] += 1

        self.accuracy = self.accuracy / y_pred.shape[0]
        print("accuracy is: ", self.accuracy)
        
        for i in range(self.model.number_of_classes):
            cur = self.labeled_confusion_matrix[i]
            cur["tn"] = y_pred.shape[0] - (cur["tp"] + cur["fn"] + cur["fp"]) #get the rest in true negative

        self.scores = {i:{"precision":0, "recall":0, "f1":0, "f2": 0} for i in range(self.model.number_of_classes)}
        for i in range(self.model.number_of_classes):
            cur_scores = self.scores[i]
            cur_mat = self.labeled_confusion_matrix[i]
            cur_scores["precision"] = cur_mat["tp"] /(cur_mat["tp"] + cur_mat["fp"]) if (cur_mat["tp"] + cur_mat["fp"]) > 0 else 0
            cur_scores["recall"] = cur_mat["tp"] / (cur_mat["tp"] + cur_mat["fn"]) if  (cur_mat["tp"] + cur_mat["fn"]) > 0 else 0
            cur_scores["f1"] = (2 * cur_scores["precision"] * cur_scores["recall"]) / (cur_scores["precision"] + cur_scores["recall"]) if (cur_scores["precision"] + cur_scores["recall"]) > 0 else 0
            cur_scores["f2"] = (5 * cur_scores["precision"] * cur_scores["recall"]) / (4*cur_scores["precision"] + cur_scores["recall"]) if (4*cur_scores["precision"] + cur_scores["recall"])> 0 else 0 
        
        print("scores")
        print(self.scores)
        print("confusion matrix")
        print(self.confusion_matrix)