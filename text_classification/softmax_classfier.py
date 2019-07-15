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
        :param params: list of parameter array
        :param lr: learning rate
        :param gradient: gradient
        :return:
        """
        return [(param - lr * gradient_vec) for param, gradient_vec in zip(params, gradient)]
    

class SGD(Optimizer):
    def __call__(self, data: np.ndarray, params: np.ndarray, gradient_func):
        np.random.shuffle(data)
        for sample in data:
            gradient = gradient_func(sample, params)
            params = self.gradient_descent(params, self.lr, gradient)
        return params
    
    
class SoftmaxClassifier:
    """
    Softmax classifier
    
    input x: vector, length is inout_dim
    z: output of the input layer, equals [weight dot x + bias]
    a: output of the classifier, a = softmax(z)
    
    """
    def __init__(self, input_dim: tuple or int, num_classes: int, seed=1):
        """
        Random initialize parameters of the model
        :param input_shape:
        :param num_classes:
        :param seed:
        """
        np.random.seed(seed)
        self.weight = np.random.rand(input_dim*num_classes).reshape([num_classes, input_dim]),
        self.bias = np.random.rand(num_classes)
    
    @staticmethod
    def _batch_softmax(z):
        """
        softmax(z) = exp(z) / sum(exp(z)
        :param z: batch of Vector, often the output of last layer
        :return: batch of Vector after softmax
        """
        exp_z = np.exp(z)
        return np.divide(exp_z, np.expand_dims(np.sum(exp_z, 1), 1))
    
    def _batch_input_out(self, batch_x):
        return np.vstack([np.dot(self.weight, x) + self.bias for x in batch_x])
    
    def gradient_function(self, example):
        """
        Calculate the gradient vector according to input example (x, y pair)
        :param example: tuple or list of x, y pair
        :return: gradient vector
        """
        batch_x, batch_y = example
        batch_a = self._batch_input_out(batch_x)
        # Partial derivative Cost -> bias
        partial_cost_partial_bias = batch_a - batch_y
        
        # Partial derivative Cost -> weight
        batch_a_minus_y = batch_a - batch_y
        partial_cost_partial_weitht = \
            np.array([np.tensordot(minus, x, axes=0) for minus, x in zip(batch_a_minus_y, batch_x)]).\
            mean(axis=0)
        
        return partial_cost_partial_weitht, partial_cost_partial_bias
        
    
if __name__ == '__main__':
    pass
