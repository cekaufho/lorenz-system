import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from pylab import xlabel, ylabel, title
import matplotlib.pyplot as pl

# x is proportional to convective intensity
# y is related to the temperature difference between descending and ascending currents
# z is the difference in vertical temperature profile from linearity in the systems of equations

#%% Parameters
sigma = 10.
b     = (8./3.)
r     = 28.
dt    = 0.01
tmax  = 5000.0

# Initial conditions
x0 = 0.0
y0 = 1e-9
z0 = 0.0

x1 = 10.0
y1 = 10.0
z1 = 10.0

t = 0.

# Prepare empty arrays
X0, X1, Y0, Y1, Z0, Z1, T = [], [], [], [], [], [], []
X0.append(x0), Y0.append(y0), Z0.append(z0), T.append(t)
X1.append(x1), Y1.append(y1), Z1.append(z1), T.append(t)

#%% Functions

def x_dot(x, y):
    return sigma*(y-x)

def y_dot(x, y, z):
    return r*x - x*z - y

def z_dot(x, y, z):
    return x*y - b*z

#%% Main for loop

while t <= tmax:
    
    # Update values for r0
    x0 += x_dot(x0, y0)*dt
    y0 += y_dot(x0, y0, z0)*dt
    z0 += z_dot(x0, y0, z0)*dt
    
    X0.append(x0)
    Y0.append(y0)
    Z0.append(z0)
    
    # Update values for r1
    x1 += x_dot(x1, y1)*dt
    y1 += y_dot(x1, y1, z1)*dt
    z1 += z_dot(x1, y1, z1)*dt
    
    X1.append(x1)
    Y1.append(y1)
    Z1.append(z1)
    
    # Update time
    t += dt
    T.append(t)

# Calculate exponents
if r >= 1:
    Cp = np.sqrt(b*(r-1))
    
#%% Plot

# Plot 3D
mpl.rcParams['legend.fontsize'] = 10
fig = pl.figure(figsize=(8,5), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X-axis', fontsize=10)
ax.set_ylabel('Y-axis', fontsize=10)
ax.set_zlabel('Z-axis', fontsize=10)
ax.plot(X0, Y0, Z0, ':', color="r", label = "Line 1 with initial condition (x0, y0, z0) = (%1.1f, %1.1f, %1.1f)" %(X0[0], Y0[0], Z0[0]))
ax.plot(X1, Y1, Z1, ':', color="b", label = "Line 2 with initial condition (x1, y1, z1) = (%1.1f, %1.1f, %1.1f)" %(X1[0], Y1[0], Z1[0]))
ax.scatter(Cp, Cp, (r-1), s=500, color="k", marker="*", label="Fixed point")
pl.title(r"Euler's Method for the Lorenz Equations where r = %1.1f" %r, fontsize=12)
ax.legend(loc='lower left')
pl.show()

# Plot XY Projection
pl.figure(figsize=(4,4))
pl.plot(X1, Y1, 'r')
pl.axis('off')
pl.show()

#%% Calculate Lyampnov Exponents

print("Fixed points, C+ =", Cp, Cp, (r-1))

L, I = [], []
K3 = np.array([X1[0], Y1[0], Z1[0]])
K4 = np.array([X0[0], Y0[0], Z0[0]])

pl.figure(figsize=(10,5))
pl.xlabel("i", fontsize=15)
pl.ylabel("L", fontsize=15)
pl.title("Lyapunov Distance of Trajectories)", fontsize=15)
    
for i in range(1, int(tmax)):   
    K1 = np.array([X1[i], Y1[i], Z1[i]])
    K2 = np.array([X0[i], Y0[i], Z0[i]])
    
    I.append(i)
    L.append(np.linalg.norm(K1-K2)/np.linalg.norm(K3-K4))
    
pl.plot(I, np.log(L), 'r')
pl.grid()
pl.show()