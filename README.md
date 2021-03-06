Finding the right architecture when applying a neural network to some classification problem is an expensive and time consuming endeavor. In this repository we use MPI4PY 
(a python library for using MPI) to parallelize the testing of various neural network architectures (hyperparameter tuning). 


Description of the code base:
1. digits_MPI.py base code that holds the parallel neural network 
2. TorchBaseline.ipynb python notebook that holds the base code for the Torch neural network
3. ArchTests contains all of the neural networks that will be tested in digits_mpi.py
4. TrainedModels contains the trained networks after digits_MPI.py is run

To run the code, download all of the necessary python libaries (mpi4py, torch, and torchvision) then run the following command to test out 4 neural network architectures in parallel: 
```
mpiexec -n 4 python digits_MPI.py
```

FUTURE: test data parallelism and splitting a single neural network architecture across processors
