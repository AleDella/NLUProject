import numpy as np
from torch.utils import data
# Set seed such that we always get the same dataset
np.random.seed(42)
# Generate the sample used for testing the network
def generate_dataset(num_sequences=100):
    samples = []    
    for _ in range(num_sequences): 
        num_tokens = np.random.randint(1, 10)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)
    return samples
# Function that prepares the dictionaries for the OH encoding
def sequences_to_dicts(sequences):
   words = set()
   for sentence in sequences:
       for word in sentence:
           words.add(word)
   words.add('UNK')
   i = 0
   idx2word = {}
   word2idx = {}
   #Convert the set into a sorted list
   words = sorted(list(words))
   for elem in words:
       word2idx[elem] = i
       idx2word[i] = elem
       i+=1
   return word2idx, idx2word, len(sequences), len(word2idx)
# Class of the dataset
class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]
        return X, y
# Creation of the actual dataset doing the training-testing-val split (80-10-10)
def create_datasets(sequences, dataset_class, p_train=0.8, p_val=0.1, p_test=0.1):
    # Define partition sizes
    num_train = int(len(sequences)*p_train)
    num_val = int(len(sequences)*p_val)
    num_test = int(len(sequences)*p_test)
    # Split sequences into partitions
    sequences_train = sequences[:num_train]
    sequences_val = sequences[num_train:num_train+num_val]
    sequences_test = sequences[-num_test:]
    # Support function: gets both targets and inputs given a sentence
    def get_inputs_targets_from_sequences(sequences):
        # Define empty lists
        inputs, targets = [], []
        # Creation of the targets (sentence shifted one to right) and 
        # inputs (sentence with one word missing)
        for sequence in sequences:
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])
         
        return inputs, targets

    # Get inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_sequences(sequences_train)
    inputs_val, targets_val = get_inputs_targets_from_sequences(sequences_val)
    inputs_test, targets_test = get_inputs_targets_from_sequences(sequences_test)
    # Create the actual datasets
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set

