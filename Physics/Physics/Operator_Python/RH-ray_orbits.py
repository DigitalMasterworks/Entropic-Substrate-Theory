
import numpy as np

def gradS(S, h):
 gx = np.zeros_like(S); gy = np.zeros_like(S)
 gx[1:-1,:] = (S[2:,:]-S[:-2,:])/(2*h)
 gy[:,1:-1] = (S[:,2:]-S[:,:-2])/(2*h)
 return gx, gy

def trace_ray(S, Lbox=100.0, steps=20000, htime=0.02, x0=(-40,0), v0=(1,0)):
 N = S.shape[0]
 h = Lbox/(N-1)
 gx, gy = gradS(S, h)
 X = []
 x, y = x0
 vx, vy = v0
 for _ in range(steps):

 i = int((x + Lbox/2)/h); j = int((y + Lbox/2)/h)
 if i<1 or i>=N-1 or j<1 or j>=N-1: break
 Sij = S[i,j]

 gxij, gyij = gx[i,j], gy[i,j]
 nx, ny = (-2.0*gxij/(Sij**3), -2.0*gyij/(Sij**3))

 ax, ay = nx, ny
 vx += htime*ax; vy += htime*ay
 vnorm = (vx*vx+vy*vy)**0.5 + 1e-12
 vx, vy = vx/vnorm, vy/vnorm
 x += htime*vx; y += htime*vy
 X.append((x,y))
 return np.array(X)

if __name__=="__main__":
 S = np.load("S_halo.npy")
 pts = trace_ray(S)
 print(f"traced {len(pts)} steps")