
import numpy as np

def halo_S(N=128, L=100.0):

 S0 = 0.8968170670639414
 Score = 0.22309312365146727
 Shalo = 0.9956807477353172
 rcore = 8.327195999840875
 rin = 12.791490262237398
 rout = 66.59679707295717


 x = np.linspace(-L/2, L/2, N)
 y = np.linspace(-L/2, L/2, N)
 X, Y = np.meshgrid(x, y, indexing='ij')
 r = np.hypot(X, Y)

 S = np.full((N,N), S0, dtype=np.float64)
 S[r <= rcore] = Score

 mask = (r > rin) & (r <= rout)
 frac = (r[mask] - rin) / (rout - rin)
 S[mask] = S0 + (Shalo - S0) * np.cos(np.pi * frac)**2
 return S

def void_S(N=128, L=100.0):

 Sbg = 0.75
 Svoid = 0.985
 R = 44.57953001819358
 sigma = 4.245098536820368

 x = np.linspace(-L/2, L/2, N)
 y = np.linspace(-L/2, L/2, N)
 X, Y = np.meshgrid(x, y, indexing='ij')
 r = np.hypot(X, Y)

 S = np.full((N,N), Sbg, dtype=np.float64)
 S[r <= R] = Svoid
 wall = (r > R)
 S[wall] += 0.18 * np.exp(-((r[wall] - R)/sigma)**2)
 return S

def aniso_S(N=128, L=100.0, shift=20.0):

 x = np.linspace(-L/2, L/2, N)
 y = np.linspace(-L/2, L/2, N)
 X, Y = np.meshgrid(x, y, indexing='ij')

 S = 0.9 + 0.08 * np.exp(-((X-shift)**2 + Y**2)/(2*(L/6)**2))
 S = np.clip(S, 0.1, 0.999)
 return S

if __name__ == "__main__":
 np.save("S_halo.npy", halo_S())
 np.save("S_void.npy", void_S())
 np.save("S_aniso.npy", aniso_S())
 print("Saved: S_halo.npy, S_void.npy, S_aniso.npy")