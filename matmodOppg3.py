import numpy as np
import matplotlib.pyplot as plt

# Define input parameters as dict on the following form:
parameters = {"PS0": 341.3,
          "CC": 0.66,
          "rSM": 0.1065,
          "rSC": 0.22,
          "rSE": 0.17,
          "aO3": 0.08,
          "aSC": 0.1239,
          "aSW": 0.1451,
          "rLC": 0.195,
          "rLE": 0,
          "aLC": 0.622,
          "aLW": 0.8258,
          "epsE": 1,
          "epsA": 0.875,
          "fA": 0.618,
          "alpha": 3,
          "beta": 4,
          "sigma": 5.67e-8}


def getSWPowers(params):
    """
    Calculate constant sw powers between earth and atmosphere/clouds
    """
    # Define needed parameters
    rSE = params["rSE"]
    rSC = params["rSC"]
    CC = params["CC"]
    aSC = params["aSC"]
    rSM = params["rSM"]
    PS0 = params["PS0"]
    aO3 = params["aO3"]
    aSW = params["aSW"]
    
    # Problem matrix
    A = np.array([[1, -rSE, 0, 0],
                  [-rSC * CC, 1, 0, -(1 - CC * rSC) * (1 - CC * aSC)],
                  [-(1 - CC * rSC) * (1 - CC * aSC), 0, 1, - rSC * CC],
                  [0, 0, -rSM, 1]])
    
    # Problem vector
    b = np.array([0,
                  0,
                  0,
                  PS0 * (1 - aO3) * (1 - rSM) * (1 - aSW)])
    
    # Solve for sw powers
    P = np.linalg.solve(A, b)
    
    # Numerical problems?
    # print(np.linalg.det(A))
    
    return P[0], P[1], P[2], P[3]



def getf(params):
    """
    For Newton scheme: get function f(T)
    """
    
    # Define needed parameters
    fA = params["fA"]
    CC = params["CC"]
    rLC = params["rLC"]
    rSC = params["rSC"]
    aSC = params["aSC"]
    aO3 = params["aO3"]
    aSW = params["aSW"]
    aLC = params["aLC"]
    aLW = params["aLW"]
    PS0 = params["PS0"]
    rSM = params["rSM"]
    epsA = params["epsA"]
    epsE = params["epsE"]
    beta = params["beta"]
    alpha = params["alpha"]
    sigma = params["sigma"]
    
    # Get SW constant powers
    PSEC, PSCE, PSCA, PSAC = getSWPowers(params)
    
    # Find composite parameters
    pi = PSCE - PSEC
    
    rho1E = epsE * (rLC * CC - 1) * sigma
    rho1A = fA * epsA * sigma
    
    K = ((PSEC + PSAC) * (1 - CC * rSC) * aSC * CC + aO3 * (PS0 + PSCA * 
         (1 - rSM) * (1 - aSW)) + aSW * (PSCA + PS0 * (1-aO3) * (1 - rSM)))
    
    rho2E = epsE * sigma * (1 - CC * rLC) * (aLC * CC + aLW * (1 - rSM) * (1 - aLC * CC))
    rho2A = -epsA * sigma
    
    gamma = alpha + beta
    
    # Define return function
    def f(T):
        
        TE = T[0]
        TA = T[1]
        
        f1 = pi + rho1A * TA ** 4 + rho1E * TE ** 4 - gamma * (TE - TA)
        f2 = K  + rho2A * TA ** 4 + rho2E * TE ** 4 + gamma * (TE - TA)
        
        f = np.array([f1,
                      f2])
        
        return f
    
    return f


def getInvJacobian(params):
    """
    For Newton scheme: get function inverse jacobian matrix of T
    """
    
    # Define needed parameters
    fA = params["fA"]
    epsE = params["epsE"]
    epsA = params["epsA"]
    sigma = params["sigma"]
    rLC = params["rLC"]
    CC = params["CC"]
    aLC = params["aLC"]
    rSM = params["rSM"]
    aLW = params["aLW"]
    alpha = params["alpha"]
    beta = params["beta"]
    
    # Composite parameters
    phi1E = 4 * epsE * sigma * (1 - rLC * CC)
    phi2E = 4 * epsE * sigma * (1 - rLC * CC) * (aLC * CC + (1 - rSM) * (1- aLC * CC) * aLW)
    gamma = alpha + beta
    
    # Define returned function
    def invJacobian(T):
        TE = T[0]
        TA = T[1]
        
        a = - phi1E * TE ** 3 - gamma
        b = 4 * fA * epsA * sigma * TA ** 3 + gamma
        c = phi2E * TE ** 3 + gamma
        d = - 4 * epsA * sigma * TA ** 3 - gamma
        
        det = a * d - b * c
        
        invJacobian = np.array([[d, -b],
                               [-c,  a]]) / det
        
        return invJacobian
    
    return invJacobian



def findTemperatureRoots(params, IC, breakCondition=0.001, maxIterations=10000):
    """
    Computes the roots of the energy flow equilibrium equations using 
    Newtons method.
    Input: model parameters as struct (in the order specified in project
    describtion), initial conditions as numpy array with two elements,
    break condition for size of iteration change, max iterations before
    break.
    """
    # Set initial condition
    Ti = IC
    
    # Define functions for updating T
    invJac = getInvJacobian(params)
    f = getf(params)
    
    iterate = True
    counter = 0
    
    # Main loop
    while iterate:
        counter += 1
        
        # Update T
        Tdiff = invJac(Ti) @ f(Ti)
        Ti = Ti - Tdiff
        
        # Check for breaking
        # Convergence 
        if np.linalg.norm(Tdiff) < breakCondition:
            iterate = False
        # Divergence
        elif counter >= maxIterations:
            iterate = False
            
    return Ti


parameters["CC"] = 0.66
T = findTemperatureRoots(parameters, [100, 470])
print("With clouds: ", T)

parameters["CC"] = 0
T = findTemperatureRoots(parameters, [290, 270])
print("Without clouds: ", T)


def centralDifferenceDerivative(params, param, eps):
    
    theta = params[param]
    
    deltaTheta = eps * theta
    
    params[param] = theta + deltaTheta
    TPlus = findTemperatureRoots(params, (300, 300))
    
    params[param] = theta - deltaTheta
    TMinus = findTemperatureRoots(params, (300, 300))
    
    TDerivative = (TPlus - TMinus) / (2 * deltaTheta)
    
    return TDerivative


TDerivative = centralDifferenceDerivative(parameters, "alpha", 1e-5)
print(TDerivative)





















"""
parameters["CC"] == 0
T = findTemperatureRoots(params, [290, 270]) 
print("Without clouds: ", T)



"""

"""
f = getf(params)

x, y = np.meshgrid(np.linspace(270, 330, 100), np.linspace(270, 330, 100))
fT = np.zeros_like(x)

for i in range(100):
    for j in range(100):
        fT[i,j] = f((x[i,j], y[i,j]))[1]

plt.imshow(fT)
plt.colorbar()
plt.show()
"""





