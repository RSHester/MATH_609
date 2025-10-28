import numpy as np

def main():

    A = createA(16)
    b = createB(A)
    x = np.zeros_like(b,dtype=np.float64)
    print("n = 16")
    GradientDescent(A,b,x)
    ConjugateGradient(A,b,x)

    A = createA(32)
    b = createB(A)
    x = np.zeros_like(b,dtype=np.float64)
    print("n = 32")
    GradientDescent(A,b,x)
    ConjugateGradient(A,b,x)

    A = np.array([[10,1,2,3,4],
                  [1,9,-1,2,-3],
                  [2,-1,7,3,-5],
                  [3,2,3,12,-1],
                  [4,-3,-5,-1,15]],dtype=np.float64)
    b = np.array([12,-27,14,-17,12],dtype=np.float64)
    x = np.zeros_like(b,dtype=np.float64)
    Jacobi(A,b,x, epsilon=1e-5)
    x = np.zeros_like(b,dtype=np.float64)
    GS(A,b,x,epsilon=1e-5)
    x = np.zeros_like(b,dtype=np.float64)
    GradientDescent(A,b,x, epsilon=1e-5)
    x = np.zeros_like(b,dtype=np.float64)
    ConjugateGradient(A,b,x)





def createA(n: int) -> np.ndarray:
    A = np.zeros((n,n),dtype=np.float64)
    for i in range(n):
        for j in range(n):
            A[i,j] = 1/(3+i+j)
    return A

def createB(A: np.ndarray) -> np.ndarray:
    b = np.sum(A,axis=1)/3
    return b

def Jacobi(A: np.ndarray, b: np.ndarray, x: np.ndarray, epsilon:float = 0.01, max_iter = 1000):
    residual = np.linalg.norm(b-A@x, np.inf)
    i = 0
    while(residual > epsilon and i <= max_iter):
        prev_residual = residual
        x = JacobiIteration(A,b,x)
        residual = np.linalg.norm(b-A@x, np.inf)
        i+=1
    print("Jacobi")
    print(x)
    print(i)
    print(residual)

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

def GS(A: np.ndarray, b: np.ndarray, x: np.ndarray, epsilon:float = 0.01, max_iter = 1000):
    residual = np.linalg.norm(b-A@x, np.inf)
    i = 0
    while(residual > epsilon and i <= 1000):
        prev_residual = residual
        x = GuassSeidelIteration(A,b,x)
        residual = np.linalg.norm(b-A@x, np.inf)
        i+=1
    print("GuassSeidel")
    print(x)
    print(i)
    print(residual)

def GuassSeidelIteration(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    for i in range(len(A)):
        summation = 0
        for j in range(len(A)):
            if i == j:
                continue
            summation +=  A[i,j]*x[j]
        x[i] = (b[i] - summation)/A[i,i]

    return x


def GradientDescent(A: np.ndarray, b: np.ndarray, x: np.ndarray, epsilon: float = 0.00001, max_iter: int = 1000):
    
    r = b-A@x
    n = 0
    print(np.linalg.norm(r,np.inf))
    while(np.linalg.norm(r,np.inf) > epsilon and n < max_iter):
        d = np.dot(r,r)/np.dot(A@r,r)
        x = x + d*r
        r = b - A@x
        n += 1
    print("GD")
    print(x)
    print(n)
    print(r)

def ConjugateGradient(A: np.ndarray, b: np.ndarray, x: np.ndarray):
    r = b-A@x
    d = -r
    n = 0
    while(n < A.shape[0]):
        z = A@d
        alpha = np.dot(r,d)/np.dot(d,z)
        x = x + alpha*d
        r = r - alpha * z
        beta = np.dot(r,z)/np.dot(d,z)
        d = -r + beta * d
        
        n += 1
    print("CG")
    print(x)
    print(n)
    print(r)


if __name__ == "__main__":
    main()