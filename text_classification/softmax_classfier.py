from loader import TrainDataSet, TestDataSet
import numpy as np


# train_dataset = TrainDataSet(r'data/raw/train.tsv')
# test_dataset = TestDataSet(r'data/raw/test.tsv')
#
# train_feature = train_dataset.bow_feature
# train_label = train_dataset.labels
#
# test_feature = test_dataset.bow_feature


def sigmoid(x: np.ndarray):
    """
    Sigmoid function:
        sigmoid(x) = 1 / (1 + exp(-x))
    
    :param x: array in
    :return: array after the calculation of sigmoid
    
    Example:
        >>>import numpy as np
        >>>arr = np.array([1, 2, 3])
        >>>sigmoid(arr)
        [0.73105858 0.88079708 0.95257413]
    """
    return 1 / (1 + np.exp(-x))


class Optimizer:
    def __init__(self, lr):
        self.lr = lr
    
    @staticmethod
    def gradient_descent(params, lr, gradient):
        """
        Gradient Descent of Single Example
        :param params: parameter array
        :param lr: learning rate
        :param gradient: gradient
        :return:
        """
        return params - lr * gradient
    

class SGD(Optimizer):
    def __call__(self, data: np.ndarray, params: np.ndarray, gradient_func):
        np.random.shuffle(data)
        for sample in data:
            gradient = gradient_func(sample, params)
            params = self.gradient_descent(params, self.lr, gradient)
        return params
    
    
class Softmax:
    def __init__(self):
        pass
    
    @staticmethod
    def cross_entropy(example, params):
        pass


if __name__ == '__main__':
    pass
