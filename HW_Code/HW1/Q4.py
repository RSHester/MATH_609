import matplotlib.pyplot as plt

epsilon = .001
n_ = [1,2,3,4,5,6,7,8]

def main():
    #TODO: Remove this line you bum
    # n_ = [1]
    
    fig1, axs1 = plt.subplots(2,2)
    fig2, axs2 = plt.subplots(2,2)
    
    for n in n_:
        #Construct lists for A B and C and f

        N = 2**n-1

        h = 2**(-n)
        A = [] 
        B = []
        C = []
        f =[]
        omega = []
        L = []
        i = 0

        while i < N:
            i+=1
            A.append((2*epsilon + h**2)/h**2)
            f.append(2*i*h+1)
        #only generate B and C if we arent 1x1
        if n > 1:
            i = 0
            while i < N:
                B.append(-epsilon/h**2)
                C.append(-epsilon/h**2)
                i+=1  

        
            #Need to get w and L
            #now we iterate until we are at the end of the list
        i=0
        while i < N:
            if i == 0:
                omega.append(A[i])
            elif i == N-1:
                L.append(C[i-1]/omega[i-1]) #Gives Li (assume 0 based indexing because memes)
                omega.append(A[i]) #This is the last step, wn = an
            else:
                L.append(C[i-1]/omega[i-1]) #Gives Li (assume 0 based indexing because memes)
                omega.append(A[i]-L[i-1]*B[i-1]) #Some tomfoolery here, L is one shorter than omega so L[i] is technically L[i-1] in storage
            i+=1


        y = []
        #now solve the equation starting with Ly=c

        #Start with y1 = f1
        y.append(f[0])
        i=1
        while i < N:
            y.append(f[i]-L[i-1]*y[i-1])
            i+=1
            
        #now solve Au=y
        u=[]
        #start with un = yn/wn
        u.append(y[-1]/omega[-1])
        i = N-2
        while i >= 0:
            u.insert(0,(y[i]-B[i]*u[0])/omega[i])
            i-=1
        print(u)

        x = []
        i = 0
        while i < N:
            x.append(h*(i+1))
            i+=1
        
        match n:
            case 1:
                axs1[0,0].plot(x,u, marker = 'o')
                axs1[0,0].set_title(f"n={n}")
            case 2:
                axs1[0,1].plot(x,u, marker = 'o')
                axs1[0,1].set_title(f"n={n}")
            case 3:
                axs1[1,0].plot(x,u, marker = 'o')
                axs1[1,0].set_title(f"n={n}")
            case 4:
                axs1[1,1].plot(x,u, marker = 'o')
                axs1[1,1].set_title(f"n={n}")
            case 5:
                axs2[0,0].plot(x,u, marker = 'o')
                axs2[0,0].set_title(f"n={n}")
            case 6:
                axs2[0,1].plot(x,u, marker = 'o')
                axs2[0,1].set_title(f"n={n}")
            case 7:
                axs2[1,0].plot(x,u, marker = 'o')
                axs2[1,0].set_title(f"n={n}")
            case 8:
                axs2[1,1].plot(x,u, marker = 'o')
                axs2[1,1].set_title(f"n={n}")

    fig1.suptitle("Question 4, n = 1-4")
    fig2.suptitle("Question 4, n = 5-8")
    plt.show()






    



if __name__ == "__main__":
    main()
