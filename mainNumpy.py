import numpy as np
from numpy_lstm.modelNumpy import Lstm
from numpy_lstm.utilsNumpy import one_hot_encode_sequence, toydataset
import matplotlib.pyplot as plt
from nltk.corpus import brown
from numpy_lstm.datasetNumpy import sequences_to_dicts, create_datasets, Dataset
'''
    This is the main in which you can see the testing of the LSTM
    implemented with python. All the files are contained in the
    "numpy_lstm" folder
'''
# create the toydataset to test the network
word2idx, idx2word, training_set, validation_set, test_set = toydataset()

# Hyperparams
hidden_size = 50 # Number of dimensions in the hidden state
vocab_size  = len(word2idx) # Size of the vocabulary used
num_epochs = 30
z_size = hidden_size + vocab_size

# Intialization
lstm = Lstm(vocab_size, hidden_size, z_size)

# Track loss
training_loss, validation_loss = lstm.train(training_set, validation_set, num_epochs, False, word2idx)

# Get first sentence in test set
inputs, targets = test_set[1]
# One-hot encode input and target sequence
inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size, word2idx)
# Forward pass
outputs = lstm.forward(inputs_one_hot)


# Print example
print('Input sentence:')
print(inputs)
print('\nTarget sequence:')
print(targets)
final = inputs
final.append(idx2word[np.argmax(outputs[-1])])
print('\nPredicted sequence:')
print(final)

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()
