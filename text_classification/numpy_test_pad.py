import numpy as np

weights = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
batch_x = np.array([[1, 2, 3, 4], [2, 2, 2, 2]])
batch_y = np.array([[1, 2, 1], [2, 3, 2]])
bias = np.array([1, 1, 1])
a = np.vstack([np.dot(weights, x)+bias for x in batch_x])
# print(a)
a_minus_y = a - batch_y
print(batch_x)
print(a_minus_y)

ret = np.array([np.tensordot(minus, x, axes=0) for minus, x in zip(a_minus_y, batch_x)])
print(ret)
print(ret.mean(axis=0))



