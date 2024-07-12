from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

executor = MPIPoolExecutor(max_workers=3)
iterable = ((2, n) for n in range(32)) # creates a iterable of tuples [(2, 0), (2, 1), ... (2, 32)]
# Does same thing as map.py but passes in the iterable of tuples directly as 1 parameter
for result in executor.starmap(pow, iterable): 
    print(result)

