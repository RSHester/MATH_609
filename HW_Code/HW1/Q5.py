import numpy as np

A1 = np.array([[2, -1, 0, 0],
              [-1, 2, -1, 0],
              [0, -1, 2, -1],
              [0, 0, -1, 2]])
A2 = np.array([[1, 1/2, 1/3, 1/4],
              [1/2, 1/3, 1/4, 1/5],
              [1/3, 1/4, 1/5, 1/6],
              [1/4, 1/5, 1/6, 1/7]])

#Cholesky is lower trianguar, can set all the upper stuff to 0
A = A2
#First calc L11 and set rest of row to 0
L = np.zeros((4,4))
L[0,0] = np.sqrt(A[0,0])

#Now do first col
for i in range(1,4):
    L[i,0] = A[i,0]/L[0,0]

#Outer for loop fills the diag, inner for loop fills the off diag elements
for i in range(1,3):
    #Get the Lik summation
    Lsum = 0
    for k in range(0,i):
        Lsum += L[i,k]**2
    L[i,i] = np.sqrt(A[i,i] - Lsum)

    #Fill in the column under the diag term
    for j in range(i+1, 4):
        #Get sum of product of L's
        Lsum = 0
        for k in range(0, i):
            Lsum += L[j,k]*L[i,k]
        L[j,i] = (A[j,i] - Lsum)/L[i,i]

#Now do the final entry
Lsum = 0
for k in range(0,4):
    Lsum += L[3,k]**2
L[3,3] = np.sqrt(A[3,3]-Lsum)

#used to check if algorithm works properly
# print(np.linalg.cholesky(A))



print(L)
