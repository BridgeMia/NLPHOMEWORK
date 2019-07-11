import pandas as pd
import numpy as np
from tqdm import tqdm


class DataSet:
    def __init__(self, fn):
        self.raw_data = pd.read_csv(fn, sep='\t')
        self.words = self._words()
        
    @staticmethod
    def _words():
        return {row['word']: row['index'] for index, row in pd.read_csv(r'data/words.txt', sep='\t').iterrows()}
        
    @property
    def bow_feature(self):
        raw_xs = self.raw_data['Phrase'].apply(lambda x: list(x.split(' '))).values
        bow_array = np.zeros((len(raw_xs), len(self.words)), dtype='int')
        for row in tqdm(range(len(raw_xs))):
            for word in raw_xs[row]:
                if self.words.get(word, None) is not None:
                    bow_array[row, self.words.get(word)] += 1
        return bow_array
    

class TrainDataSet(DataSet):
    @property
    def labels(self):
        return self.raw_data['Sentiment'].values
    
    
class TestDataSet(DataSet):
    pass


if __name__ == '__main__':
    ds = TrainDataSet(r'data/raw/train.tsv')
    print(ds.bow_feature.shape)
    print(ds.labels.shape)
