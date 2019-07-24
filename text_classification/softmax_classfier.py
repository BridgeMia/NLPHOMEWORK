from tqdm import tqdm

from loader import TrainDataSet, TestDataSet
import numpy as np
from utils import index_divider
from optimizer import optimizers, SGD, Optimizer


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
        self.weight = np.random.rand(input_dim*num_classes).reshape([num_classes, input_dim])
        self.bias = np.random.rand(num_classes)
        self.optimizer = None
        self.lr = None
        self.compiled = False
        self.history = {
            'epoch': [],
            'loss': [],
            'acc': [],
            'current_loss': [],
            'current_acc': []
        }
        
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
        return self._batch_softmax(np.vstack([np.dot(self.weight, x) + self.bias for x in batch_x]))
    
    def get_gradient(self, example):
        """
        Calculate the gradient vector according to input example (x, y pair)
        :param example: tuple or list of x, y pair
        :return: gradient vector
        """
        batch_x, batch_y = example
        batch_a = self._batch_input_out(batch_x)
        # Partial derivative Cost -> bias
        partial_cost_partial_bias = (batch_a - batch_y).mean(axis=0)
        
        # Partial derivative Cost -> weight
        batch_a_minus_y = batch_a - batch_y
        partial_cost_partial_weight = \
            np.array([np.tensordot(minus, x, axes=0) for minus, x in zip(batch_a_minus_y, batch_x)]).\
            mean(axis=0)
        
        return partial_cost_partial_weight, partial_cost_partial_bias
    
    def compile(self, optimizer: str or Optimizer, lr=None):
        if isinstance(optimizer, str):
            self.optimizer = optimizers[optimizer](lr)
        else:
            self.optimizer = optimizer
        self.compiled = True
        
    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int, epoch: int):
        if not self.compiled:
            raise RuntimeError("Compile the model before fitting")
        
        data_size = len(x)
        data_index = np.arange(data_size)
        for _ in range(epoch):
            print("Epoch: %d" % (_ + 1))
            np.random.shuffle(data_index)
            batch_indexes = index_divider(data_index, batch_size, data_size)
            t_iter = tqdm(batch_indexes)
            for batch_index in t_iter:
                batch_x = x[batch_index]
                batch_y = y[batch_index]
                weight_gradient, bias_gradient = self.get_gradient((batch_x, batch_y))
                self.weight, self.bias = self.optimizer([self.weight, self.bias], [weight_gradient, bias_gradient])
                batch_out = self._batch_input_out(batch_x)
                batch_loss = -(np.multiply(batch_y, np.log(batch_out))).mean()
                self.history['current_loss'].append(batch_loss)
                batch_accuracy = sum([a == y for a, y in zip(batch_out.argmax(1), batch_y.argmax(1))]) / len(batch_y)
                self.history['current_acc'].append(batch_accuracy)
                t_iter.set_description("Loss: %f, Accuracy: %f" % (batch_loss, batch_accuracy))
                
            loss = -(np.multiply(y, np.log(self._batch_input_out(x)))).mean()
            acc = sum([a == _y for a, _y in zip(self._batch_input_out(x).argmax(1), y.argmax(1))]) / len(y)
            
            print("Loss: %f, Accuracy: %f" % (loss, acc))
            self.history['epoch'].append(_+1)
            self.history['loss'].append(loss)
            self.history['acc'].append(acc)
      

if __name__ == '__main__':
    train_dataset = TrainDataSet(r'data/raw/train.tsv')
    test_dataset = TestDataSet(r'data/raw/test.tsv')
    
    train_feature = train_dataset.bow_feature
    train_label = train_dataset.labels
    from keras.utils import to_categorical
    train_label = to_categorical(train_label)
    test_feature = test_dataset.bow_feature
    print(train_feature.shape)
    print(train_label.shape)
    
    model = SoftmaxClassifier(train_feature.shape[1], train_label.shape[1])
    model.compile('sgd', 0.05)
    model.fit(train_feature, train_label, batch_size=256, epoch=30)
    from utils import save_list
    save_list(model.history['current_loss'], r'data/batch_loss.txt')
    save_list(model.history['current_acc'], r'data/batch_acc.txt')
    save_list(model.history['loss'], r'data/loss.txt')
    save_list(model.history['acc'], r'data/acc.txt')
