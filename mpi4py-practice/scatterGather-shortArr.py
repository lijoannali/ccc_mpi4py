import numpy as np 
import mpi4py 
from mpi4py import MPI 

#Replace thread parallelism with MPI implementation
# get number of cores to use
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
n_jobs = size 

#Lengths of local input chunk 
local_n = np.empty(1, dtype=int) 
local_n_ccc = np.empty(1, dtype=int) 


# Inputs to be scattered
inputs = np.array([1, 2], dtype=int)
inputs_ccc = np.array([3, 4], dtype=int) #hardcoded to make 2 chunks

#Allocate results array
results = None; 
results_ccc = None; 

if rank == 0: 
    #Set the size of results
    local_n = np.array([1]) #hardcoded
    local_n_ccc = np.array([1])

    #Arrays to gather results to 
    results = np.empty([2, 1], dtype=int)
    results_ccc = np.empty([2, 1], dtype=int)

#Send size = [1] to all ranks
comm.Bcast(local_n, root=0)
comm.Bcast(local_n_ccc, root=0)

#Allocate recv buffers
local_input = np.empty([1, local_n[0]], dtype=int)
local_input_ccc = np.empty([1, local_n_ccc[0]], dtype=int)
#Scatter input to procs by rank
comm.Scatter(inputs, local_input, 0)
comm.Scatter(inputs_ccc, local_input_ccc, 0)

print("Input, rank", rank, "=", local_input)
print("Input_ccc, rank", rank, "=", local_input_ccc)

#Local computation here
local_input *= 10; 
local_input_ccc *= 10; 

print("rank", rank, "local inp after comp ", local_input)
print("rank", rank, "local inpccc after comp ", local_input_ccc)

#Gather results on rank 0 
comm.Gather(local_input, results, 0)
comm.Gather(local_input_ccc, results_ccc, 0)

if rank == 0: 
    print ("Results", results)
    print ("Results_ccc", results_ccc)