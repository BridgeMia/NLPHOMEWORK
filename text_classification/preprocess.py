import os

import pandas as pd
from collections import Counter
import numpy as np
from utils import save_dict

train_dataset = pd.read_csv(r'data/raw/train.tsv', sep='\t')
phrase_set = np.hstack(train_dataset['Phrase'].apply(lambda x: list(x.split(' '))).values)

tf = Counter(phrase_set)

if not os.path.exists(r'data/tf'):
    os.mkdir(r'data/tf')
save_dict(tf, r'data/tf/tf.txt')

words = pd.DataFrame({'word': list(tf.keys())}).sample(frac=1, random_state=1)
words = words.reset_index(drop=True).reset_index()

words.to_csv(r'data/words.txt', sep='\t', index=False)
