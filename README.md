# Artificial Neural Network Discipline Labs
Welcome to the repository for labs in the Artificial Neural Network discipline. In these labs, we explore fundamental concepts in neural networks, implementing key components from scratch to deepen our understanding. The labs are structured around the following topics:

1. Activation Functions
2. Optimizers
3. Self Organizing Maps
4. Hopfield Network

## Labs Overview

1. ### Activation Functions

Implement various activation functions from scratch.
Explore the impact of different activation functions on network performance.
Understand the role of activation functions in neural network architectures.

2. ### Optimizers

Implement a simple autograd system from scratch.
Develop custom optimization algorithms.
Investigate the behavior of different optimizers during training.

3. ### Self Organizing Maps

Build and analyze Self Organizing Maps (SOMs).
Understand the principles of unsupervised learning in the context of SOMs.
Visualize and interpret the learned representations.

4. ### Hopfield Network

Implement the Hopfield Network for associative memory.
Explore the capabilities and limitations of Hopfield Networks.
Analyze the network's ability to recall stored patterns.
Implementation Details
For the first two labs (Activation Functions and Optimizers), a simple autograd system has been implemented from scratch. The core idea is based on a base interaction class called Tensor. This class can be manipulated through a variety of methods. Each method is represented by a specific class that tracks every change in memory. This ensures that every operation has a unique state and can be reproduced for backpropagation during the training process.
