from math import exp
import numpy as np
import os
import matplotlib.pyplot as plt

class nonlinear_least_squares:
    def __init__(self, func, x):
        self.func = func
        self.x = x
        self.y = [func.get_y(x_i) for x_i in x]
        if not os.path.isdir("graph"):
            os.mkdir("graph")

    def Newton(self, param):
        zero = np.array([0]*len(self.x))
        temp_p = np.array(param)
        temp_p.shape = 1,-1
        temp_p = temp_p.T
        i = 0
        while(True):
            i+=1
            help_p = [temp_p[0][0],temp_p[1][0],temp_p[2][0],temp_p[3][0]]
            new_func = my_func(help_p[0], help_p[1], help_p[2], help_p[3])
            plt.plot(np.array(self.x), np.array(self.y), np.array(self.x), np.array([new_func.get_y(x_i) for x_i in self.x]))
            plt.savefig("./graph/fig"+str(i)+".png")
            plt.clf()
            J = np.array(self.get_J(help_p))
            r = np.array(self.get_r(help_p))
            print("J: \n", J)
            print("JT*J: \n", J.T.dot(J))
            print("r: \n", r)
            if (r == zero).all(): break
            r.shape = 1,-1
            first = np.linalg.inv(J.T.dot(J))
            second = first.dot(J.T).dot(r.T)
            temp_p = temp_p - (second)
            print("Param: ",temp_p)

    def get_J(self, param):
        new_func = my_func(param[0], param[1], param[2], param[3])
        J = []
        for x_i in self.x:
            J.append([-new_func.dA(x_i),-new_func.dB(x_i),-new_func.dmu(x_i),-new_func.dsigma(x_i)])
        return J
    
    def get_r(self, param):
        r = []
        new_func = my_func(param[0], param[1], param[2], param[3])
        for i in range(len(self.x)):
            r.append(self.y[i] - new_func.get_y(self.x[i]))
        return r
        
class my_func:
    def __init__(self, A, B, mu, sigma):
        self.A = A
        self.B = B
        self.mu = mu
        self.sigma = sigma
        
    def get_y(self, x):
        A = self.A
        B = self.B
        mu = self.mu
        sigma = self.sigma
        return A*exp(-(x-mu)**2/(2*sigma**2))+B

    def dA(self, x):
        B = self.B
        mu = self.mu
        sigma = self.sigma
        return exp(-(x-mu)**2/(2*sigma**2)) 

    def dB(self, x):
        result = 0
        return 1

    def dmu(self, x):
        A = self.A
        B = self.B
        mu = self.mu
        sigma = self.sigma
        return A*(x-mu)*exp(-(x-mu)**2/(2*sigma**2))/(sigma**2)

    def dsigma(self, x):
        A = self.A
        B = self.B
        mu = self.mu
        sigma = self.sigma
        return A*((x-mu)**2)*exp(-(x-mu)**2/(2*sigma**2))/(sigma**3)
        
    
f = my_func(168, 21, 183, 46)
x = []
for i in range(20):
    x.append(i*i)
mnk = nonlinear_least_squares(f, x)
mnk.Newton([30, 40, 70, 15])
