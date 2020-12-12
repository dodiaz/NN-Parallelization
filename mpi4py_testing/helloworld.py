#!/usr/bin/env python

from mpi4py import MPI
import sys

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

sys.stdout.write("Hello world! I am processor " + str(rank) + " of " + str(size) + " on " +  str(name) + ". \n") 

