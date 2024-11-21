import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy import linalg

def a(x, xs):
    au = 0.38
    al = 0.68
    return np.where(x==xs, (au+al)/2,np.where(x>xs, au, al))
def S(x):
    S2 = -0.477
    return 1+S2*(3*x**2-1)/2

Q = 1360/4 # W/m^2
Aout = 201.4 # W/m^2
Bout = 1.45 # W/m^2C
D = 0.3 # W/m^2K Can we consider Celsius?

def Q_optim(x):
    return

xs = 0.8

N = 1000
x = np.linspace(0, 1, N+1)
h = 1/N

# Upper diagonal
Ah = np.diag([2*(1-x[1]**2)]     +    [1-x[i+1]**2 for i in range(1, N)], k=1)
# Middle diagonal
Ah += np.diag([2*(x[1]**2 - h**2*Bout/D - 1)]    +     [x[i+1]**2 + x[i-1]**2 - 2*h**2*Bout/D - 2 for i in range(1,N)]  +  [2*(x[-2]**2 - h**2*Bout/D - 1)])
# Lower diagonal
Ah += np.diag([1-x[i-1]**2 for i in range(1,N)]    +     [2*(1-x[-2]**2)], k=-1)

# Alternatives
# Ah = np.diag([2*(1-h**2)]+[1-(i+1)**2*h**2 for i in range(1, N)], k=1)
# Ah += 2*np.diag([(i**2+1-Bout/D)*h**2 - 1 for i in range(N+1)])
# Ah += np.diag([1-(i-1)**2*h**2 for i in range(1,N)]+[2*(1-(N**2+1)*h**2)], k=-1)

Ah /= 2*h**2

def analyze_stability(N, Ah, D=0.3, Bout=1.45, Ca=1.0):
    """
    Analyze the stability of the system
    """
    A = Ah
    eigenvals = linalg.eigvals(A)
    is_stable = np.all(np.real(eigenvals) < 0)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(np.real(eigenvals), np.imag(eigenvals), alpha=0.6)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.title(f'Eigenvalue Spectrum (N={N})\nSystem is {"stable" if is_stable else "unstable"}')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    
    stats = f'Max Re(λ): {np.max(np.real(eigenvals)):.2e}\n'
    stats += f'Min Re(λ): {np.min(np.real(eigenvals)):.2e}'
    plt.text(0.02, 0.98, stats, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return eigenvals, is_stable

# Test the implementation0
for N in [10, 20, 50, 600]:
    print(f"\nTesting with N = {N}")
    eigenvals, is_stable = analyze_stability(N, Ah)
    max_real = np.max(np.real(eigenvals))
    print(len(np.where(np.real(eigenvals) > 0)))
    print(f"Maximum real part of eigenvalues: {max_real:.6f}")
    print(f"System is {'stable' if is_stable else 'unstable'}")

# Right hand side
# F = np.array([Aout-Q*S(xi)*a(xi, xs) for xi in x])/D
# U = np.linalg.solve(Ah, F)
# # Temperature at node closest to ice cap boundary
# print(U[np.argmin(np.abs(x-xs))])

# # Plot solution as temperature over degrees latitude
# plt.plot(np.arcsin(x)*180/np.pi, U, label='Temperature as a function of latitude')
# plt.xlabel('Latitude [degrees]')
# plt.ylabel('Temperature [C]')
# plt.xlim([0, 90])
# plt.show()