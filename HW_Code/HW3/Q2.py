import numpy as np

def main():
    A = np.array([[3, 1, 0],
                [1 ,3, 1],
                [0, 1, 3]])

    b = np.array([4,5,4],dtype=np.float64)

    U = np.zeros(A.shape,dtype=np.float64)
    D = np.zeros(A.shape,dtype=np.float64)
    L = np.zeros(A.shape,dtype=np.float64)

    #populating Arrays
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                D[i,j] = A[i,j]
            elif i < j:
                U[i,j] = A[i,j]
            elif i > j:
                L[i,j] = A[i,j]    

    Tj= - np.linalg.inv(D) @ (L + U)
    # print(Tj)
    rhoTj = np.max(np.linalg.eig(Tj)[0])
    # print(rhoTj)

    Tgs = -np.linalg.inv(L+D) @ U
    # print(Tgs)
    rhoTgs = np.max(np.linalg.eig(Tgs)[0])
    # print(rhoTgs)

    wstar = 2/(1 + np.sqrt(1-rhoTj**2))
    # print(wstar)
    xstar = np.array([1,1,1],dtype=np.float64)
    
    print("Jacobi")
    x = np.array([0,0,0],dtype=np.float64)
    prev_error = getErrorNorm(xstar,x)
    for i in range(10):
        print(f'{i} & {x[0]:.4f} & {x[1]:.4f} & {x[2]:.4f} & {getErrorNorm(xstar,x):.4f} & {getErrorNorm(xstar,x) / prev_error:.4f}\\\\')
        prev_error = getErrorNorm(xstar,x)
        x = JacobiIteration(A,b,x)
    print("Gauss-Seidel")
    x = np.array([0,0,0],dtype=np.float64)
    prev_error = getErrorNorm(xstar,x)
    for i in range(10):
        print(f'{i} & {x[0]:.4f} & {x[1]:.4f} & {x[2]:.4f} & {getErrorNorm(xstar,x):.4f} & {getErrorNorm(xstar,x) / prev_error:.4f}\\\\')
        prev_error = getErrorNorm(xstar,x)
        x = GuassSeidelIteration(A,b,x)

    print("SOR")
    x = np.array([0,0,0],dtype=np.float64)
    prev_error = getErrorNorm(xstar,x)
    for i in range(10):
        print(f'{i} & {x[0]:.4f} & {x[1]:.4f} & {x[2]:.4f} & {getErrorNorm(xstar,x):.4f} & {getErrorNorm(xstar,x) / prev_error:.4f}\\\\')
        prev_error = getErrorNorm(xstar,x)
        x = SORIteration(A,b,x,wstar)



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


def SORIteration(A: np.ndarray, b: np.ndarray, x: np.ndarray, w: float) -> np.ndarray:
    xnew = np.zeros_like(x,dtype=np.float64)
    
    for i in range(len(A)):
        summation = - b[i]
        for j in range(len(A)):
            if i == j:
                continue
            elif j < i:
                summation += A[i,j] * xnew[j]
            elif j > i:
                summation += A[i,j] * x[j]
        xnew[i] = (1-w)*x[i] - w/A[i,i] * summation

    return xnew


def getErrorNorm(xstar: np.ndarray, x: np.ndarray):
    e = abs(xstar-x)
    return (max(e))

if __name__ == "__main__":
    main()