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
inputs = np.array([0,1, 2,3 ], dtype=int)
inputs_ccc = np.array([4, 5, 6, 7], dtype=int) #hardcoded to make 2 chunks

if rank == 0: 
    #Set the size of results
    local_n = np.array([2]) #hardcoded
    local_n_ccc = np.array([2])

#Send size = [1] to all ranks
comm.Bcast(local_n, root=0)
comm.Bcast(local_n_ccc, root=0)

#Allocate recv buffers
local_input = np.empty([size, local_n[0]], dtype=int)
local_input_ccc = np.empty([size, local_n_ccc[0]], dtype=int)
#Scatter input to procs by rank
comm.Scatter(inputs, local_input, 0)
# comm.Scatter(inputs_ccc, local_input_ccc, 0)

print("Input, rank", rank, "=", local_input)
# print("Input_ccc, rank", rank, "=", local_input_ccc)


#Local computation here

#parts[local_input[0]] = compute_parts(local_input[0])

#Second local ccc computation here

#Gather results on rank 0 