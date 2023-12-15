import numpy as np

class ModelSaveLoad:
    @staticmethod
    def save(model, title):
        if model.trained:
            np.save(title, model.weights)
            print("saved")
        else:
            print("model not trained")

    @staticmethod
    def load(model, title):
        try:
            model.weights = np.load(title + ".npy")
            model.trained = True
            print("found!")
        except OSError:
            print("not found")