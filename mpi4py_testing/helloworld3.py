#!/usr/bin/env python
import sys


sys.stdout.write("Only one processor will print this (hopefully) \n")



from mpi4py import MPI

sys.stdout.write("Only one processor will print this (hopefully, test 2) \n")

MPI_Init(NULL, NULL);

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

if (rank == 0): 
    sys.stdout.write("Only one processor will print this \n")

sys.stdout.write("Hello world! I am processor " + str(rank) + " of " + str(size) + " on " +  str(name) + ". \n") 

if (rank == 0):
    sys.stdout.write("Only one processor will print this \n")
    
MPI_Finalize();
