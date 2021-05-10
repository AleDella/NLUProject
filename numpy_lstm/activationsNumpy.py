import numpy as np

def sigmoid(x):
      return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    return (np.exp(x)/sum(np.exp(x)))