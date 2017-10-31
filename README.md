This is for my research in applying CNN to solve a quantum physics problem. 

Basically, I want to use Convolutional Neural Network as a variational ansatz for ground state wavefunction.  Quite similar to image classification problems,
parameters of CNN is found by optimizing an Energy Functional. 

Difference here is that I need to use monte carlo to calculate and optimize energy. As a result, open-source package like tensorflow is not efficient.
Therefore, I implemented all of low level codes for CNN by myself, including forwarding and backwarding process.

Main codes include:

1.neural_network.h: Implement a class for CNN here.  Include both forwarding and backwarding process. Traing via monte-carlo is also a method function of this class.

2. Components of CNN: conv_layer.h, fc_layer.h, pool_layer.h etc.  These are classes for key components of CNN, like convolutional layer and pooling layer. 
Again, backwarding and forwarding are methods of each class.

