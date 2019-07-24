import numpy as np
from matplotlib import pyplot as plt
from utils import load_list

loss = load_list(r'data/loss.txt')
loss = np.array([float(x) for x in loss])

acc = load_list(r'data/acc.txt')
acc = np.array([float(x) for x in acc])

plt.plot(range(len(loss)), loss)
plt.plot(range(len(acc)), acc)

plt.show()
