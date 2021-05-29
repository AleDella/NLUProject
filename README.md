# NLUProject
Final Project for Natural Language Understanding Course 2021

This repository contains two implementation of simple LSTM for language modeling:
 the first one is done entirely with numpy as a sort of "trial" implementation in order to understand better the mechanics behind a LSTM Neural Network; the files for this implementation are in the `numpy_lstm` folder. If you want to test the implementation with a really small dataset execute `main_Numpy.py`.
 The second one (contained in the colab file) is the proper implementation tested on PTB Dataset.

Network Statistics (PyTorch):
|Phase Name|Training       | Validation    | Testing  |
| ---------|:-------------:|:-------------:| --------:|
| Loss     |4.597          |4.856          | 4.795    |
|Perplexity|99.244         |128.539        | 121.024  |
