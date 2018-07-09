# QuCUMBER
A Quantum Calculator Used for Many-Body Eigenstate Reconstruction.

QuCUMBER is a program that reconstructs an unknown quantum wavefunction
from a set of measurements.  The measurements should consist of binary counts; 
for example, the occupation of an atomic orbital, or the $S^z$ eigenvalue of
a qubit.  These measurements form a training set, which is used to train a
stochastic neural network called a Restricted Boltzmann Machine.  Once trained, the
neural network is a reconstructed representation of the unknown wavefunction
underlying the measurement data. It can be used for generative modelling, i.e.
producing new instances of measurements, and to calculate estimators not
contained in the original data set.

QuCUMBER is developed by the Perimeter Institue Quantum Intelligence Lab (PIQuIL).

## Features
QuCUMBER implements unsupervised generative modelling with a two-layer RBM.  
Each layer is a number of binary stochastic variables (0 or 1).  The size of the visible
layer corresponds to the input data; i.e. the number of qubits.  The size of the hidden
unit is varied to control representation error.

Currently the reconstruction can be performed on pure states with a positive-definite 
wavefunction.  Data is thus only required in one basis.  Upcoming versions will 
allow reconstruction of more general wavefunctions and density matrices; however 
tomographyically-complete basis sets may be required in the training data.

## Requirements
The code is written in PyTorch, with CPU and GPU support.  See https://pytorch.org.

## Compiling
## Running
