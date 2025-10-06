import numpy as np

def substrate_harmonic(grid_size=10):
 x = np.linspace(-1,1,grid_size)
 y = np.linspace(-1,1,grid_size)
 X,Y = np.meshgrid(x,y)
 R = np.sqrt(X**2+Y**2)
 S = R

 u = np.cos(np.pi*X)*np.cos(np.pi*Y)
 lap = (np.roll(u,1,0)-2*u+np.roll(u,-1,0)) \
 + (np.roll(u,1,1)-2*u+np.roll(u,-1,1))
 return np.allclose(lap,0,atol=1e-1)

print("Harmonic form closure?", substrate_harmonic())