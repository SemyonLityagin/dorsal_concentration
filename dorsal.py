from math import exp
import numpy as np
"""
def LU(W):
    U = W.copy()
    L = []
    for i in range(len(W)):
        temp = []
        for j in range(len(W)):
            temp.append(0)
        temp[i] = 1
        L.append(temp)
    
    for i in range(len(L)):
        for j in range(i+1, len(L)):
            L[j][i] = U[j][i]/U[i][i]
        for k in range(i, len(L)):
            print(U[i][i])
            U[j][k] = U[j][k] - U[i][k]*U[j][i]/U[i][i]

    for i in L:
        print(i)
    for i in U:
        print(i)
"""

class nonlinear_least_squares:
    def __init__(self, func, x):
        self.func = func
        self.x = x
        self.y = [func.get_y(x_i) for x_i in x]

    def Newton(self, param):
        zero = np.array([[self.func.A],[self.func.B],[self.func.mu],[self.func.sigma]])
        temp_p = np.array(param)
        temp_p.shape = 1,-1
        temp_p = temp_p.T
        while(not (temp_p == zero).all()):
            """
            help_p = [temp_p[0][0],temp_p[1][0],temp_p[2][0],temp_p[3][0]]
            new_func = my_func(help_p[0], help_p[1], help_p[2], help_p[3])
            W = np.array(self.get_W(help_p))
            r = np.array(self.get_E(help_p))
            print("W: ", W)
            print("r: ", r)
            r.shape = 1,-1
            first = np.linalg.inv(W)
            second = first.dot(r.T)
            temp_p = temp_p - second
            print("Param: ",temp_p)
            """
            help_p = [temp_p[0][0],temp_p[1][0],temp_p[2][0],temp_p[3][0]]
            new_func = my_func(help_p[0], help_p[1], help_p[2], help_p[3])
            J = np.array(self.get_J(help_p))
            r = np.array(self.get_r(help_p))
            print("JT*J: ", J.T.dot(J))
            print("r: ", r)
            r.shape = 1,-1
            first = np.linalg.inv(J.T.dot(J))
            second = first.dot(J.T.dot(r.T))
            temp_p = temp_p - second
            print("Param: ",temp_p)
            
            

    def var(self, param):
        new_func = my_func(param[0], param[1], param[2], param[3])
        y = [new_func.get_y(x_i) for x_i in self.x]
        summ = 0
        for y_i in y:
            summ += y_i
        average = summ/len(y)
        variance = 0
        for y_i in y:
            variance = (average - y_i)**2
        variance = variance/(len(y)-1)
        V = []
            
        for i in range(len(y)):
            temp = []
            for j in range(len(y)):
                temp.append(0)
            temp[i] = 1/variance**2
            V.append(temp)
        return V

    def get_W(self, param):
        W = []
        W.append([self.dA_dA(param), self.dA_dB(param), self.dA_dmu(param), self.dA_dsigma(param)])
        W.append([self.dA_dB(param), self.dB_dB(param), self.dB_dmu(param), self.dB_dsigma(param)])
        W.append([self.dA_dmu(param), self.dB_dmu(param), self.dmu_dmu(param), self.dmu_dsigma(param)])
        W.append([self.dA_dsigma(param), self.dB_dsigma(param), self.dmu_dsigma(param), self.dsigma_dsigma(param)])
        return W

    def get_J(self, param):
        new_func = my_func(param[0], param[1], param[2], param[3])
        J = []
        for x_i in self.x:
            J.append([-new_func.dA(x_i),-new_func.dB(x_i),-new_func.dmu(x_i),-new_func.dsigma(x_i)])
        return J
    
    def get_E(self, param):
        E_A = 0
        E_B = 0
        E_mu = 0
        E_sigma = 0
        new_func = my_func(param[0], param[1], param[2], param[3])
        for i in range(len(self.x)):
            E_A += (self.y[i] - new_func.get_y(self.x[i]))*new_func.dA(self.x[i])
            E_B += (self.y[i] - new_func.get_y(self.x[i]))*new_func.dB(self.x[i])
            E_mu += (self.y[i] - new_func.get_y(self.x[i]))*new_func.dmu(self.x[i])
            E_sigma += (self.y[i] - new_func.get_y(self.x[i]))*new_func.dsigma(self.x[i])
        return[2*E_A, 2*E_B, 2*E_mu, 2*E_sigma]

    def get_r(self, param):
        r = []
        new_func = my_func(param[0], param[1], param[2], param[3])
        for i in range(len(self.x)):
            r.append(self.y[i] - new_func.get_y(self.x[i]))
        return r
        
    def dA_dA(self, param):
        result = 0
        for x_i in self.x:
            result += exp(-((x_i-param[2])**2)/(param[3]**2))
        return result

    #dB_dB
    def dA_dB(self, param):
        result = 0
        for x_i in self.x:
            result += exp(-((x_i-param[2])**2)/(2*param[3]**2))
        return result

    #dmu_dA
    def dA_dmu(self, param):
        result = 0
        new_func = my_func(param[0], param[1], param[2], param[3])
        for i, x_i in enumerate(self.x):
            e1 = exp(-((x_i-param[2])**2)/(param[3]**2))
            e2 = exp(-((x_i-param[2])**2)/(2*param[3]**2))
            result += (param[0]*e1-(self.y[i] - new_func.get_y(x_i))*e2)*(x_i-param[2])/param[3]**2
        return result
    
    #dsigma_dA
    def dA_dsigma(self, param):
        result = 0
        new_func = my_func(param[0], param[1], param[2], param[3])
        for i, x_i in enumerate(self.x):
            e1 = exp(-((x_i-param[2])**2)/(param[3]**2))
            e2 = exp(-((x_i-param[2])**2)/(2*param[3]**2))
            result += (param[0]*e1-(self.y[i] - new_func.get_y(x_i))*e2)*((x_i-param[2])**2)/(param[3]**3)
        return result
        
    def dB_dB(self, param):
        result = 0
        for x_i in self.x:
            result += 1
        return result
        
    #dmu_dB
    def dB_dmu(self, param):
        result = 0
        new_func = my_func(param[0], param[1], param[2], param[3])
        for x_i in self.x:
            e1 = exp(-((x_i-param[2])**2)/(2*param[3]**2))
            result += param[0]*e1*(x_i-param[2])/(param[3]**2)
        return result
        
    #dsigma_dB
    def dB_dsigma(self, param):
        result = 0
        new_func = my_func(param[0], param[1], param[2], param[3])
        for x_i in self.x:
            e1 = exp(-((x_i-param[2])**2)/(2*param[3]**2))
            result += param[0]*e1*((x_i-param[2])**2)/(param[3]**3)
        return result
        
    def dmu_dmu(self, param):
        result = 0
        new_func = my_func(param[0], param[1], param[2], param[3])
        for i, x_i in enumerate(self.x):
            e1 = exp(-((x_i-param[2])**2)/(param[3]**2))
            e2 = exp(-((x_i-param[2])**2)/(2*param[3]**2))
            result += param[0]**2*e1*((x_i-param[2])**2)/(param[3]**4)
            result += (self.y[i] - new_func.get_y(x_i))*e2*param[0]*(param[3]**2-(x_i-param[2])**2)/(param[3]**4)
        return result

    #dsigma_dmu
    def dmu_dsigma(self, param):
        result = 0
        new_func = my_func(param[0], param[1], param[2], param[3])
        for i, x_i in enumerate(self.x):
            e1 = exp(-((x_i-param[2])**2)/(param[3]**2))
            e2 = exp(-((x_i-param[2])**2)/(2*param[3]**2))
            result += param[0]**2*e1*((x_i-param[2])**3)/(param[3]**5)
            result += (self.y[i] - new_func.get_y(x_i))*e2*param[0]*(x_i-param[2])*(2*param[3]**2-(x_i-param[2])**2)/(param[3]**5)
        return result
    
    def dsigma_dsigma(self, param):
        result = 0
        new_func = my_func(param[0], param[1], param[2], param[3])
        for i, x_i in enumerate(self.x):
            e1 = exp(-((x_i-param[2])**2)/(param[3]**2))
            e2 = exp(-((x_i-param[2])**2)/(2*param[3]**2))
            result += param[0]**2*e1*((x_i-param[2])**4)/(param[3]**6)
            result += (self.y[i] - new_func.get_y(x_i))*e2*param[0]*(x_i-param[2])**2*(3*param[3]**2-(x_i-param[2])**2)/(param[3]**6)
        return result

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
mnk = nonlinear_least_squares(f, [10, 2, 3, 4,5])
mnk.Newton([168, 21, 183, 80])
