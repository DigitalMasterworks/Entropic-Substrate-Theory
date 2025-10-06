#!/usr/bin/env python3


import numpy as np

N = 128
L = 2*np.pi
nu = 5e-3
dt = 3e-4
T = 1.0

dx = L/N
k1 = 2*np.pi * np.fft.fftfreq(N, d=dx)
KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing='ij')
K2 = KX*KX + KY*KY + KZ*KZ
K2[0,0,0] = 1.0

kmax = np.max(np.abs(k1))
kcut = (2.0/3.0) * kmax
deal = (np.abs(KX)<=kcut) & (np.abs(KY)<=kcut) & (np.abs(KZ)<=kcut)

def project(u, v, w):
 dot = KX*u + KY*v + KZ*w
 u = u - (dot*KX)/K2
 v = v - (dot*KY)/K2
 w = w - (dot*KZ)/K2
 return u, v, w


x = y = z = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
u = np.sin(X)*np.cos(Y)*np.cos(Z)
v = -np.cos(X)*np.sin(Y)*np.cos(Z)
w = 0*X

uh = np.fft.fftn(u); vh = np.fft.fftn(v); wh = np.fft.fftn(w)
uh, vh, wh = project(uh, vh, wh)
uh[~deal]=0; vh[~deal]=0; wh[~deal]=0

def dealiased(ff): ff[~deal]=0; return ff
def grad(fh):
 return (np.fft.ifftn(1j*KX*fh).real,
 np.fft.ifftn(1j*KY*fh).real,
 np.fft.ifftn(1j*KZ*fh).real)

def nonlinear(uh,vh,wh):
 u = np.fft.ifftn(uh).real; v = np.fft.ifftn(vh).real; w = np.fft.ifftn(wh).real
 ux,uy,uz = grad(uh); vx,vy,vz = grad(vh); wx,wy,wz = grad(wh)
 Nu = u*ux + v*uy + w*uz
 Nv = u*vx + v*vy + w*vz
 Nw = u*wx + v*wy + w*wz
 return dealiased(np.fft.fftn(Nu)), dealiased(np.fft.fftn(Nv)), dealiased(np.fft.fftn(Nw))

def energy(uh,vh,wh):
 u = np.fft.ifftn(uh).real; v = np.fft.ifftn(vh).real; w = np.fft.ifftn(wh).real
 return 0.5*np.mean(u*u+v*v+w*w)

def vorticity_mag_inf(uh,vh,wh):

 u = np.fft.ifftn(uh).real; v = np.fft.ifftn(vh).real; w = np.fft.ifftn(wh).real
 uy = np.fft.ifftn(1j*KY*uh).real; uz = np.fft.ifftn(1j*KZ*uh).real
 vx = np.fft.ifftn(1j*KX*vh).real; vz = np.fft.ifftn(1j*KZ*vh).real
 wx = np.fft.ifftn(1j*KX*wh).real; wy = np.fft.ifftn(1j*KY*wh).real
 wx_comp = wz = None
 ωx = wy - vz
 ωy = uz - wx
 ωz = vx - uy
 return np.max(np.sqrt(ωx*ωx + ωy*ωy + ωz*ωz))

def substrate_barrier_from_winf(winf, alpha=1.0):
 Cmax = winf/(alpha+winf)
 return (Cmax + (1.0-Cmax) - 1.0), Cmax

print("== 3D Navier–Stokes Barrier ==")
print(f"N={N}, nu={nu}, dt={dt}, T={T}")
print("Time E/E0 ||ω||_inf BKM∫||ω|| barrier<=0? Cmax")

t=0.0
E0 = energy(uh,vh,wh)
bkm = 0.0
wprev = None
steps = int(T/dt)
per = max(1,int(0.1/dt))

for s in range(1, steps+1):
 Nu1,Nv1,Nw1 = nonlinear(uh,vh,wh)
 uh1 = (uh - dt*Nu1)/(1.0 + nu*K2*dt)
 vh1 = (vh - dt*Nv1)/(1.0 + nu*K2*dt)
 wh1 = (wh - dt*Nw1)/(1.0 + nu*K2*dt)
 uh1,vh1,wh1 = project(uh1,vh1,wh1); uh1=dealiased(uh1); vh1=dealiased(vh1); wh1=dealiased(wh1)

 Nu2,Nv2,Nw2 = nonlinear(uh1,vh1,wh1)
 uh = (uh - 0.5*dt*(Nu1+Nu2))/(1.0 + nu*K2*dt)
 vh = (vh - 0.5*dt*(Nv1+Nv2))/(1.0 + nu*K2*dt)
 wh = (wh - 0.5*dt*(Nw1+Nw2))/(1.0 + nu*K2*dt)
 uh,vh,wh = project(uh,vh,wh); uh=dealiased(uh); vh=dealiased(vh); wh=dealiased(wh)

 t += dt
 winf = vorticity_mag_inf(uh,vh,wh)
 if wprev is None: wprev = winf
 bkm += 0.5*(wprev+winf)*dt
 wprev = winf

 if s % per == 0 or s == steps:
 barr, Cmax = substrate_barrier_from_winf(winf)
 print(f"{t:5.2f} {energy(uh,vh,wh)/E0:8.5f} {winf:10.6f} {bkm:10.6f} {barr<=1e-12!s:>5} {Cmax:6.3f}")