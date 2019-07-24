import numpy as np

__all__ = """Optimizer, SGD, optimizers"""


class Optimizer:
    def __init__(self, lr):
        self.lr = lr
    
    def __call__(self, params: np.ndarray, gradient):
        pass
    

class SGD(Optimizer):
    @staticmethod
    def gradient_descent(params, lr, gradient):
        """
        Gradient Descent of Single Example
        :param params: list of parameter array
        :param lr: learning rate
        :param gradient: gradient
        :return:
        """
        return [(param - lr * gradient_vec) for param, gradient_vec in zip(params, gradient)]
    
    def __call__(self, params: np.ndarray, gradient):
        return self.gradient_descent(params, self.lr, gradient)


optimizers = {
    'sgd': SGD,
    'SGD': SGD
}
