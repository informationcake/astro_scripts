import multiprocessing
import itertools
import numpy as np

def some_function(n, constant):
    print('calculating {1:.2f}*{0:.2f}^2'.format(n, constant))
    f = constant*n**2
    print('done')
    return f

data=np.arange(1,100)
constant=8

#find out how many processes are available
n_proc=multiprocessing.cpu_count()
#run some function in parallel
with multiprocessing.Pool(processes=n_proc) as pool:
        result=pool.starmap(some_function, zip(data, itertools.repeat(constant)))
        pool.close()

#Notice how multiprocessing does the 100 lines of the data array in batches of n_proc.

print(result)
#note that all the arguments in zip() need to be the same size. So the data array contains lots of different entries where each is run separetly, but itertools.repeat passes the same constant to each run so you don't have to have an array the same size as data but only containing the same constant over and over again.
