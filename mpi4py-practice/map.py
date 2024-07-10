from mpi4py import MPI 
from mpi4py.futures import MPIPoolExecutor 

if __name__ == '__main__':
    executor = MPIPoolExecutor(max_workers=3)
    # Applies pow function to the array of 32 '2's and raises them to the power of 0, 1, ..., 32 
    #result = elements of [2^0, 2^1, ... , 2^32]
    for result in executor.map(pow,[2]*32, range(32)): 
        print (result)

