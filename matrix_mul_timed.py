import time
import numpy as np

a = [[2,3,4,5,4,3,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [8,9,10,43,2,5,2,3,4,5,4,3,2,3,4,5,4,3],
     [2,3,4,5,4,3,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [8,9,10,43,2,5,2,3,4,5,4,3,2,3,4,5,4,3],
     [2,3,4,5,4,3,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [5,6,7,45,34,2,2,3,4,5,4,3,2,3,4,5,4,3],
     [8,9,10,43,2,5,2,3,4,5,4,3,2,3,4,5,4,3]]

def matrix_mul(A, B):
    if len(A[0]) != len(B):
        return -1
    a_dim = (len(A), len(A[0]))
    b_dim = (len(B), len(B[0]))
    matrix = [[0 for x in range(a_dim[1])] for y in range(b_dim[0])]
    for x in range(a_dim[0]):
        for y in range(b_dim[1]):
            matrix[x][y] = sum([a[x][z] * b[z][y] for z in range(a_dim[0])])
    return matrix

startM = time.time() # Timing my algorithm
resultM = matrix_mul(a,b)
endM = time.time()

print(resultM)

totalM = endM - startM
print("Total Time For My Alg: " + str(totalM))

b = np.array(b)
a = np.array(a)

startN = time.time() # Timing the GPU numpy algorithm
resultN = a.dot(b)
endN = time.time()

print(resultN)

totalN = endN - startN
print("Total Time For NUMPY: " + str(totalN))
