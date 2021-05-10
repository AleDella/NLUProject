import numpy as np
from .datasetNumpy import generate_dataset, sequences_to_dicts, create_datasets, Dataset

# Create a toy dataset in order to check functionalities
def toydataset():
    # In this case the dataset will be composed by sentences formed by 
    # 'a a a a b b EOS' and phrases like that
    sequences = generate_dataset()
    # Retrieve the dictionaries for the encoding
    word2idx, idx2word, num_sequences, vocab_size = sequences_to_dicts(sequences)    
    #Creates the three partitions of the dataset
    training_set, validation_set, test_set = create_datasets(sequences, Dataset)

    return word2idx, idx2word, training_set, validation_set, test_set


# Function that encodes a single word given index and vocab_size
def one_hot_encode(idx, vocab_size):
    # Initialize the encoded array
    one_hot = np.zeros(vocab_size)
    # Set the appropriate element to one
    one_hot[idx] = 1.0
    return one_hot
# Function that encodes the whole sequences
def one_hot_encode_sequence(sequence, vocab_size, word2idx):
    # Encode each word in the sentence
    encoding = np.array([one_hot_encode(word2idx[word], vocab_size) for word in sequence])
    # Reshape encoding s.t. it has shape (num words, vocab size, 1)
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)
    return encoding
# Functions that initialize the weights of the model
def init_orthogonal(param):
    """
    Initializes weight parameters orthogonally.
    
    Refer to this paper for an explanation of this initialization:
    https://arxiv.org/abs/1312.6120
    """
    if param.ndim < 2:
        raise ValueError("Only parameters with 2 or more dimensions are supported.")
    rows, cols = param.shape
    new_param = np.random.randn(rows, cols)
    if rows < cols:
        new_param = new_param.T
    # Compute QR factorization
    q, r = np.linalg.qr(new_param)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph
    if rows < cols:
        q = q.T
    new_param = q
    return new_param
# Clips gradient to a maximum in order to avoid exploding gradients problem
def clip_gradient_norm(grads, max_norm=0.25):
    # Set the maximum of the norm to be of type float
    max_norm = float(max_norm)
    total_norm = 0
    # Calculate the L2 norm squared for each gradient and add them to the total norm
    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm
    total_norm = np.sqrt(total_norm)
    # Calculate clipping coeficient
    clip_coef = max_norm / (total_norm + 1e-6)
    # If the total norm is larger than the maximum allowable norm, then clip the gradient
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef
    return grads
