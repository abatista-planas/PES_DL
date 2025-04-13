"""module containing a predictor which makes predictions"""
import pickle
import os



class Predictor:
    """make predictions from a model"""

    def __init__(self, path: str):
        """initialize class from path to artifact named clf.pickle

        The pickled model must hvae a predict method which accepts
        lists of lists of 4 floats

        """
        with open(os.path.join(path, "clf.pickle"), "rb") as f:
            self.__model = pickle.load(f)

    @property
    def model(self):
        """the model used to make predictions"""
        return self.__model

    def predict_one(self, x: list[float]) -> int:
        """take list of length four and return class label 0, 1, 2

        Arguments:
            x (list[float]): list of floats of length four

        Returns:
            class label 0, 1, 2

        """
        return self.model.predict([x])[0]

    def predict_batch(self, x: list[list[float]]) -> list[int]:
        """take lists of lists and make preductions

        Arguments:
            x (list[list[float]]): list of lists of length four

        Returns:
            list of classlables 0, 1, 2

        """
        return list(self.model.predict(x))
