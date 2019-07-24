# Text Classification Based on Softmax & SGD

Record how the classifier works

## Input and Output

The original dataset is text and label, from [Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

## Preprocess

We use Bag-of-Word to represent the Text, the size of the feature vector will depend on the vocabulary size of the train dataset.

In this representation method, the we can assume each word in the vocabulary is the index of the feature vector, the length of the vector $n$ is the number of distinct words in train text. The value in the index of feature vector is the number of times the word turns up in the text

For example, we now have a train set of two texts:

```python
train_set = [
    "I love NLP", 
    "NLP is interesting"
]
```

The dictionary of word and index might be:

```python
word_index = {
    "I": 0,
    "love": 1, 
    "NLP": 2,
    "is": 3, 
    "interesting": 4
}
```

The texts in the train data set could be represented as :

```python
train_set_vectors = [
    [1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1]
]
```

When we have a new text as prediction/test dataset:

```python
test_set = ["I am learning NLP"]
test_vectors = [[1, 0, 1, 0, 0]]
```



Stop words and punctuations are reserved since this mission is more about how SGD works rather than the classification accuracy.

Label is represented as one-hot vector.

## Mathematics

We assume the single example is a x-y pair, where $x$ is the bag-of-word representation of the text, which is an 1-d array, and $y$ is a one-hot vector, which is also an 1-d array.

The output of the model is the probability distribution of each tag