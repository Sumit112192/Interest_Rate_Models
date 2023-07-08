import numpy as np
import scipy
import matplotlib.pyplot as plt


#The below reads the rates from file fedRates.dat
#Simply after this line, you would be having a 1D list of rates
rates = np.array([float(line.strip().split()[4])/100 for line in open("../fedRates.dat").readlines()])

dt = 1/12 #Since the spot rates are 1 month apart.

Y = []
Z = []
for i in range(len(rates)-1):
    Y.append(rates[i+1]-rates[i]) #Assuming positive interest rates
    Z.append( [dt , dt*rates[i]])


theta = scipy.linalg.lstsq(Z, Y)
N = len(rates)
alpha = -theta[0][1]
mu = theta[0][0]/alpha
res = theta[1]
sigma = np.sqrt(res)/np.sqrt(dt*N)
print("OLS Results: ")
print("Alpha = ", alpha, "Mu = ", mu, "sigma = ", sigma)

def VASlog(guess):
    alpha = guess[0]
    mu = guess[1]
    sigma = guess[2]
    c = sigma**2*(1-np.exp(-2*alpha*dt))/(2*alpha)
    u = np.exp(-alpha*dt)*rates[:-1]
    v = rates[1:]
    q = mu*(1-np.exp(-alpha*dt))

    logL = -(np.sum(np.log(2*np.pi*c)+(v-u-q)**2/(c)))/2
    return -logL

optimizedParam = scipy.optimize.fmin(VASlog, (alpha, mu, sigma))
print("Maximum Likelihood Estimation Result: ")
print(optimizedParam)




