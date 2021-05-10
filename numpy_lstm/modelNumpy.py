import numpy as np
from .utilsNumpy import init_orthogonal, clip_gradient_norm, one_hot_encode_sequence
from .activationsNumpy import tanh, sigmoid, softmax
# Implementation of a LSTM cell using numpy
class LstmCell():
    # vocab_size --> input size
    # hidden_size --> hidden state size
    # c_size --> output size (cell state size)
    def __init__(self, vocab_size, hidden_size, c_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.c_size = c_size
        # Weight matrix (forget gate)
        w_forget = np.random.randn(hidden_size, c_size)
        # Bias for forget gate
        self.b_forget = np.zeros((hidden_size, 1))
        # Weight matrix (input gate)
        w_input = np.random.randn(hidden_size, c_size)
        # Bias for input gate
        self.b_input = np.zeros((hidden_size, 1))
        # Weight matrix (candidate)
        w_intermediate = np.random.randn(hidden_size, c_size)
        # Bias for candidate
        self.b_intermediate = np.zeros((hidden_size, 1))
        # Weight matrix of the output gate
        w_output = np.random.randn(hidden_size, c_size)
        self.b_output = np.zeros((hidden_size, 1))
        # Weight matrix relating the hidden-state to the output
        w_final = np.random.randn(vocab_size, hidden_size)
        self.b_final = np.zeros((vocab_size, 1))
        # Initialize weights according to https://arxiv.org/abs/1312.6120
        self.w_forget = init_orthogonal(w_forget)
        self.w_input = init_orthogonal(w_input)
        self.w_intermediate = init_orthogonal(w_intermediate)
        self.w_output = init_orthogonal(w_output)
        self.w_final = init_orthogonal(w_final)
    # Forward pass of the network
    # Input:
    #   + sequence in input
    #   + h_prev --> previous hidden state (short-term memory)
    #   + c_prev --> previous cell state (long-term memory)
    # Return:
    #   + all the lists for the parameters
    def forward(self, inputs, h_prev, c_prev):
        # Check on the shapes of the states
        assert h_prev.shape == (self.hidden_size, 1)
        assert c_prev.shape == (self.hidden_size, 1)
        # Save a list of computations for each of the components in the LSTM
        # (One list for component like frg_s = forget saves)
        stacked_s, frg_s, inpt_s  = [], [] ,[]
        int_s, cell_s, otpt_s, hidden_s = [], [] ,[], []
        final_s, output_s =  [], [] 
        # Append the initial cell and hidden state to their respective lists
        hidden_s.append(h_prev)
        cell_s.append(c_prev)
        # Process the sequence
        for x in inputs:
            # Concatenate input and hidden state
            # (convenient for future computations)
            stacked = np.row_stack((h_prev, x))
            stacked_s.append(stacked)
            # Calculate forget gate
            frg = sigmoid(np.dot(self.w_forget, stacked) + self.b_forget)
            frg_s.append(frg) 
            # Calculate input gate
            inpt = sigmoid(np.dot(self.w_input, stacked) + self.b_input)
            inpt_s.append(inpt)
            # Calculate intermediate cell state
            inter = tanh(np.dot(self.w_intermediate, stacked) + self.b_intermediate)
            int_s.append(inter)
            # Calculate memory state
            c_prev = frg * c_prev + inpt * inter 
            cell_s.append(c_prev)
            # Calculate output gate
            otpt = sigmoid(np.dot(self.w_output, stacked) + self.b_output)
            otpt_s.append(otpt)
            # Calculate hidden state
            h_prev = otpt * tanh(c_prev)
            hidden_s.append(h_prev)
            # Calculate logits
            final = np.dot(self.w_final, h_prev) + self.b_final
            final_s.append(final)
            # Calculate softmax
            output = softmax(final)
            output_s.append(output)
        return stacked_s, frg_s, inpt_s, int_s, cell_s, otpt_s, hidden_s, final_s, output_s
    # Backward pass (i.e. loss and gradient computations)
    # procedure taken from https://cs231n.github.io/neural-networks-case-study/#grad
    # Input:
    #   + the lists of the parameters given by forward()
    #   + list of outputs
    #   + list of targets
    # Return:
    #   + the loss
    #   + the list of the updated gradients for all the parameters
    def backward(self, stacked_s, frg_s, inpt_s, int_s, cell_s, otpt_s, hidden_s, fin_s, outputs, targets):
        # Initialize gradients as zero
        # Forget gate
        w_f_grad = np.zeros_like(self.w_forget)
        b_f_grad = np.zeros_like(self.b_forget)
        # Input gate
        w_i_grad = np.zeros_like(self.w_input)
        b_i_grad = np.zeros_like(self.b_input)
        # Cell state
        w_int_grad = np.zeros_like(self.w_intermediate)
        b_int_grad = np.zeros_like(self.b_intermediate)
        # Output gate
        w_o_grad = np.zeros_like(self.w_output)
        b_o_grad = np.zeros_like(self.b_output)
        # Final processing
        w_fin_grad = np.zeros_like(self.w_final)
        b_fin_grad = np.zeros_like(self.b_final)
        # Set the next cell and hidden state equal to zero
        dh_next = np.zeros_like(hidden_s[0])
        dc_next = np.zeros_like(cell_s[0])
        # Track loss
        loss = 0
        for t in reversed(range(len(outputs))):
            # Compute the cross entropy
            loss += -np.mean(np.log(outputs[t]) * targets[t])
            # Get the previous hidden cell state
            c_prev= cell_s[t-1]
            # Compute the derivative of the relation of the hidden-state to the output gate
            dfin = np.copy(outputs[t])
            dfin[np.argmax(targets[t])] -= 1
            # Update the gradient of the relation of the hidden-state to the output gate
            w_fin_grad += np.dot(dfin, hidden_s[t].T)
            b_fin_grad += dfin
            # Compute the derivative of the hidden state and output gate
            dh = np.dot(self.w_final.T, dfin)        
            dh += dh_next
            do = dh * tanh(cell_s[t])
            do = (1-(sigmoid(otpt_s[t])**2))*do
            # Update the gradients with respect to the output gate
            w_o_grad += np.dot(do, stacked_s[t].T)
            b_o_grad += do
            # Compute the derivative of the cell state and candidate g
            dc = np.copy(dc_next)
            dc += dh * otpt_s[t] * (1-(tanh(tanh(cell_s[t]))**2))
            dint = dc * inpt_s[t]
            dint = (1-(tanh(int_s[t])**2)) * dint
            # Update the gradients with respect to the candidate
            w_int_grad += np.dot(dint, stacked_s[t].T)
            b_int_grad += dint
            # Compute the derivative of the input gate and update its gradients
            di = dc * int_s[t]
            di = (1-(sigmoid(inpt_s[t])**2)) * di
            w_i_grad += np.dot(di, stacked_s[t].T)
            b_i_grad += di
            # Compute the derivative of the forget gate and update its gradients
            df = dc * c_prev
            df = sigmoid(frg_s[t]) * df
            w_f_grad += np.dot(df, stacked_s[t].T)
            b_f_grad += df
            # Compute the derivative of the input and update the gradients of the previous hidden and cell state
            dz = (np.dot(self.w_forget.T, df)
                + np.dot(self.w_input.T, di)
                + np.dot(self.w_intermediate.T, dint)
                + np.dot(self.w_output.T, do))
            dh_prev = dz[:self.hidden_size, :]
            dc_prev = frg_s[t] * dc
        grads= w_f_grad, w_i_grad, w_int_grad, w_o_grad, w_fin_grad, b_f_grad, b_i_grad, b_int_grad, b_o_grad, b_fin_grad
        # Clip gradients
        grads = clip_gradient_norm(grads)
        return loss, grads

    # Function that does gradient descent and updates the values of the cell
    # Input:
    #   + list of gradients for the parameters (returned by backward())
    #   + learning rate (lr)
    # Returns:
    #   nothing
    def update_parameters(self, grads, lr=1e-3):
        self.w_forget -= lr * grads[0]
        self.w_input -= lr * grads[1]
        self.w_intermediate -= lr * grads[2]
        self.w_output -= lr * grads[3]
        self.w_final -= lr * grads[4]
        self.b_forget -= lr * grads[5]
        self.b_input -= lr * grads[6]
        self.b_intermediate -= lr * grads[7]
        self.b_output -= lr * grads[8]
        self.b_final -= lr * grads[9]

# Implementation of the actual LSTM network
class Lstm():

    def __init__(self, vocab_size, hidden_size, c_size):
      self.vocab_size = vocab_size
      self.hidden_size = hidden_size
      self.c_size = c_size
      self.cell = LstmCell(vocab_size, hidden_size, c_size)
    # Function that does the forward pass of the network
    # Inputs:
    #   + list of inputs
    #   + hidden state (h)
    #   + cell state (c)
    #   + targets --> if target label is given
    #   + train --> if is doing training or not
    # Return:
    #   + loss or list of outputs depending on the value of "train"
    def forward(self, inputs, h=None, c=None, targets = None, train = False):
        # Initialize the previous hidden state and cell state
        if(h is None):
            h = np.zeros((self.hidden_size, 1))
        if(c is None):
            c = np.zeros((self.hidden_size, 1))
        # Forward pass of the cell
        stacked_s, frg_s, inpt_s, int_s, cell_s, otpt_s, hidden_s, final_s, outputs = self.cell.forward(inputs, h, c)
        # This serie of if-else distinguish between the training phase,
        # the validation phase or the normal forward of the net
        if(targets is not None):
            loss, grads = self.cell.backward(stacked_s, frg_s, inpt_s, int_s, cell_s, otpt_s, hidden_s, final_s, outputs, targets)
            if(train == True):
                self.cell.update_parameters(grads, lr=1e-1)
            return loss
        else:
            return outputs
    # This is the function that does all the training
    # Inputs:
    #   + training set
    #   + validation set
    #   + number of epochs
    #   + embedded --> set to True if the sets are already embedded
    #   + embedding --> the dictionary used for embedding (default method One-Hot encoding)
    # Return:
    #   + training losses and validation losses through the training
    def train(self, training_set, validation_set, epochs, embedded=True, embedding=None):
        # Track loss
        training_loss, validation_loss = [], []
        # For each epoch
        for i in range(epochs):
            # Track loss
            epoch_training_loss = 0
            epoch_validation_loss = 0
            # For each sentence in validation set
            for inputs, targets in validation_set:
                # One-hot encode input and target sequence if not already embedded
                if(not embedded):
                    inputs_one_hot = one_hot_encode_sequence(inputs, self.vocab_size, embedding)
                    targets_one_hot = one_hot_encode_sequence(targets, self.vocab_size, embedding)
                else:
                    inputs_one_hot = inputs
                    targets_one_hot = targets
                # Calculate the loss
                loss = self.forward(inputs_one_hot, targets=targets_one_hot)
                epoch_validation_loss += loss
            # For each sentence in training set
            for inputs, targets in training_set:
                # One-hot encode input and target sequence
                if(not embedded):
                    inputs_one_hot = one_hot_encode_sequence(inputs, self.vocab_size, embedding)
                    targets_one_hot = one_hot_encode_sequence(targets, self.vocab_size, embedding)
                else:
                    inputs_one_hot = inputs
                    targets_one_hot = targets
                # Initialize hidden state and cell state as zeros
                loss = self.forward(inputs_one_hot, targets=targets_one_hot, train=True)
                # Update loss
                epoch_training_loss += loss
            # Save loss for plot
            training_loss.append(epoch_training_loss/len(training_set))
            validation_loss.append(epoch_validation_loss/len(validation_set))
            # Print loss every 5 epochs
            if i % 5 == 0:
                print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')
        return training_loss, validation_loss