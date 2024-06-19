from mpi4py import MPI 
import numpy as np

comm = MPI.COMM_WORLD 
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters for this script
# cols = 2
# nrows = 2
# rows = np.array([1, 2, 3], dtype=np.int32) # why is this int32? This works as long as length = ranks
#rows = nrows * np.ones((size), dtype=np.int32) # makes an array of 2s of length size, 2 = height of rows
#rows[-1] = 1 #What is the point of this line?

# Construct some data
# data = [np.array((), dtype=np.double) for _ in range(size)] #data is an list of arrays of length size 
# data[rank] = np.array(rank+np.random.rand(rows[rank], cols), np.double) # Fills data in with random [cols x nrows] 2D arrays
data1 = np.array([[1, 2, 3], [4, 5, 6]], dtype= np.double)
data2 = np.array([[7, 8, 9], [10, 11, 12]], dtype= np.double)
data = [data1, data2] 

# Compute rows and offsets for Allgatherv
rows_memory = 6 # Calculates num of elts in each array of data 
sendcounts = [6, 6]
offsets = [0,6]
print(offsets)

# if rank == 0:
#    print(f"Total rows {np.sum(rows)}")
#    print(f"rows: {rows}")
#    print(f"array sizes: {rows_memory}")
#    print(f"Offsets: {offsets}")

# Prepare buffer for Allgatherv
data_out = np.empty((2, 2, 3), dtype=np.double)
comm.Allgather(
   data[rank],
   recvbuf=[data_out, sendcounts, offsets, MPI.DOUBLE])

if (rank == 0):
   print(f"Data_out has shape {data_out.shape}")
   print(data_out)
