How to run the helloworld.py file in gcp (I might have missed one or two steps but I think these are the most important ones) which uses mpi4py to print "hello world" to the command line:

```python
sudo apt-get install build-essential
sudo apt-get install git google-perftools  

git clone https://github.com/dodiaz/NN-Parallelization.git
git checkout Dominic
sudo apt-get install openmpi-bin
sudo apt-get install python
sudo apt-get install python-pip
pip install mpi4py

cd mpi4py_testing
mpiexec -n 4 python helloworld.py
```

Note: you might want to try this on one of the more expensive instance types that have access to many nodes. I had initially used a E2-small instance but was only able to start 1 MPI process. When I switched to the E2-standard-8 instance, I was able to start 4 MPI processes.
