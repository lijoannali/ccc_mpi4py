from mpi4py import MPI 
from mpi4py.futures import MPIPoolExecutor

if __name__ == '__main__':
    executor = MPIPoolExecutor(max_workers=1)
    future = executor.submit(pow, 2, 3) #function, param1, param2, etc; this returns 2^3
    print(future.result())

# submit(func, *args, **kwargs)
# func is executed as func(*args, **kwargs)- kwargs is keyword args
# future is a Future object, you call Future.result() to get the value of the executed function