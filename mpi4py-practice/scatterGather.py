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
local_n = np.zeros(1, dtype=int) 
local_n_ccc = np.zeros(1, dtype=int) 

#Allocate recv buffers
local_input = np.ndarray((1), dtype=int)
local_input_ccc = np.ndarray((1), dtype=int) 

# Inputs to be scattered
inputs = (np.array([0]), np.array([1]))
inputs_ccc = (np.array([0])) #hardcoded to make 2 chunks
print("Inputs", inputs)
print("Inputs_ccc", inputs_ccc)

#Pad an input, just for testing on 2 procs
if (len(inputs_ccc) == 1): 
    inputs_ccc.append(np.array([1]))
    print("type is", type(inputs_ccc), len(inputs_ccc))

if rank == 0: 
    #Allocate results array
    local_n = np.array([1]) #hardcoded to length of 1
    local_n_ccc = np.array([1])

#Send size = [1] to all ranks
comm.Bcast(local_n, root=0)
comm.Bcast(local_n_ccc, root=0)

#All ranks: 
#Scatter input to procs by rank
comm.Scatter(inputs, local_input, 0)
comm.Scatter(inputs_ccc, local_input_ccc, 0)

print("Input, rank", rank, "=", local_input)
print("Input_ccc, rank", rank, "=", local_input)


#Local computation here

#parts[local_input[0]] = compute_parts(local_input[0])

#Second local ccc computation here

#Gather results on rank 0 