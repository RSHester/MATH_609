import numpy as np


def main():
    A = np.array([[4,-1,0,-1,0,0,0,0,0],
              [-1,4,-1,0,-1,0,0,0,0],
              [0,-1,4,0,0,-1,0,0,0],
              [-1,0,0,4,-1,0,-1,0,0],
              [0,-1,0,-1,4,-1,0,-1,0],
              [0,0,-1,0,-1,4,0,0,-1],
              [0,0,0,-1,0,0,4,-1,0],
              [0,0,0,0,-1,0,-1,4,-1],
              [0,0,0,0,0,-1,0,-1,4]],
              dtype=np.float64)
    
    b = np.array([0,0,1,0,0,1,0,0,1],dtype = np.float64)

    #Jacobi
    x = np.zeros_like(b,dtype=np.float64)
    prev_residual = 0
    residual = calcResidual(A,b,x)
    i = 0
    while(residual > .01 and i <= 100):
        prev_residual = residual
        x = JacobiIteration(A,b,x)
        residual = calcResidual(A,b,x)
        i+=1
    print(f'{i-1} & {residual:.4f} & {residual/prev_residual:.4f}')

    #Gauss-Seidel
    x = np.zeros_like(b,dtype=np.float64)
    prev_residual = 0
    residual = calcResidual(A,b,x)
    i = 0
    while(residual > .01 and i <= 100):
        prev_residual = residual
        x = GuassSeidelIteration(A,b,x)
        residual = calcResidual(A,b,x)
        i+=1
    print(f'{i-1} & {residual:.4f} & {residual/prev_residual:.4f}')





def JacobiIteration(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    xnew = np.zeros_like(x,dtype=np.float64)

    for i in range(len(A)):
        summation = 0
        for j in range(len(A)):
            if i == j:
                continue
            summation +=  A[i,j]*x[j]
        xnew[i] = (b[i] - summation)/A[i,i]

    return xnew


def GuassSeidelIteration(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    for i in range(len(A)):
        summation = 0
        for j in range(len(A)):
            if i == j:
                continue
            summation +=  A[i,j]*x[j]
        x[i] = (b[i] - summation)/A[i,i]

    return x

def calcResidual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:

    r = np.zeros_like(x,dtype=np.float64)

    r += b.T - A@x.T

    return max(abs(r))




if __name__ == "__main__":
    main()