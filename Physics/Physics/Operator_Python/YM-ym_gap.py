
import argparse, math
import numpy as np

try:
 import scipy.sparse as sp
 import scipy.sparse.linalg as spla
 HAVE_SCIPY = True
except Exception:
 HAVE_SCIPY = False

def su2_random(beta, rng):

 sigma = 1.0/max(np.sqrt(beta), 1e-6)
 theta = rng.normal(0.0, sigma)
 n = rng.normal(size=3); n /= (np.linalg.norm(n) + 1e-12)
 a = math.cos(theta)
 b, c, d = n*math.sin(theta)

 U = np.array([[a+1j*d, 1j*b + c],
 [1j*b - c, a-1j*d]], dtype=np.complex128)

 U, _ = np.linalg.qr(U)
 if np.linalg.det(U) < 0:
 U[:,0] *= -1
 return U

def idx3d_to_lin(x, y, z, L):
 return (x % L) + L*((y % L) + L*(z % L))

def build_links(L, beta, rng):

 Ux = np.empty((L, L, L, 2, 2), dtype=np.complex128)
 Uy = np.empty((L, L, L, 2, 2), dtype=np.complex128)
 Uz = np.empty((L, L, L, 2, 2), dtype=np.complex128)
 for z in range(L):
 for y in range(L):
 for x in range(L):
 Ux[x,y,z] = su2_random(beta, rng)
 Uy[x,y,z] = su2_random(beta, rng)
 Uz[x,y,z] = su2_random(beta, rng)
 return Ux, Uy, Uz

def assemble_H(L, Ux, Uy, Uz, use_sparse=True):

 nsite = L*L*L
 dim = 2*nsite
 if use_sparse:
 from scipy.sparse import lil_matrix
 H = lil_matrix((dim, dim), dtype=np.complex128)
 else:
 H = np.zeros((dim, dim), dtype=np.complex128)

 def add_block(i, j, B):
 if use_sparse:
 H[i:i+2, j:j+2] = H[i:i+2, j:j+2] + B
 else:
 H[i:i+2, j:j+2] += B

 I2 = np.eye(2, dtype=np.complex128)

 for z in range(L):
 for y in range(L):
 for x in range(L):
 s = idx3d_to_lin(x,y,z,L)
 i0 = 2*s


 add_block(i0, i0, 6.0*I2)


 nx = idx3d_to_lin(x+1,y,z,L); j0 = 2*nx
 add_block(i0, j0, -Ux[x,y,z])

 px = idx3d_to_lin(x-1,y,z,L); pj0 = 2*px
 add_block(i0, pj0, -Ux[x-1,y,z].conj().T)


 ny = idx3d_to_lin(x,y+1,z,L); j0 = 2*ny
 add_block(i0, j0, -Uy[x,y,z])

 py = idx3d_to_lin(x,y-1,z,L); pj0 = 2*py
 add_block(i0, pj0, -Uy[x,y-1,z].conj().T)


 nz = idx3d_to_lin(x,y,z+1,L); j0 = 2*nz
 add_block(i0, j0, -Uz[x,y,z])

 pz = idx3d_to_lin(x,y,z-1,L); pj0 = 2*pz
 add_block(i0, pj0, -Uz[x,y,z-1].conj().T)

 if use_sparse:
 H = H.tocsr()

 H = (H + (H.conj().T)) * 0.5
 return H

def smallest_eigs(H, k=4):
 dim = H.shape[0]
 if HAVE_SCIPY and hasattr(H, "tocsr"):
 vals = spla.eigsh(H, k=min(k, dim-2), which="SA", return_eigenvectors=False)
 vals = np.sort(vals.real)
 return vals

 w = np.linalg.eigvalsh(H.toarray() if hasattr(H, "toarray") else H)
 return np.sort(w.real)

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--L", type=int, default=10)
 ap.add_argument("--beta", type=float, default=4.0)
 ap.add_argument("--eigs", type=int, default=4)
 ap.add_argument("--configs", type=int, default=5)
 ap.add_argument("--seed", type=int, default=0)
 ap.add_argument("--N", type=int, default=32, help="Grid size (number of modes per dimension)")
 ap.add_argument("--nu", type=float, default=0.01, help="Viscosity coefficient")
 ap.add_argument("--dt", type=float, default=1e-3, help="Time step")
 ap.add_argument("--T", type=float, default=0.5, help="Total integration time")
 ap.add_argument("--ic", type=str, default="taylor-green",
 choices=["taylor-green", "random"], help="Initial condition")
 args = ap.parse_args()

 rng = np.random.default_rng(args.seed)
 lam_mins = []
 for c in range(args.configs):
 Ux, Uy, Uz = build_links(args.L, args.beta, rng)
 use_sparse = HAVE_SCIPY
 H = assemble_H(args.L, Ux, Uy, Uz, use_sparse=use_sparse)
 vals = smallest_eigs(H, k=args.eigs)

 eps = 1e-10
 nz = vals[vals > eps]
 lam_min = float(nz[0]) if nz.size else float(vals[-1])
 lam_mins.append(lam_min)
 print(f"[cfg {c+1}/{args.configs}] smallest nonzero ≈ {lam_min:.6g}")

 lam_mins = np.array(lam_mins)
 mean = float(np.mean(lam_mins))
 se = float(lam_mins.std(ddof=1)/np.sqrt(max(1,len(lam_mins)-1)))
 print(f"[YM-gap] L={args.L}, beta={args.beta}, configs={args.configs} → "
 f"⟨λ_min⟩ = {mean:.6g} ± {se:.2g}")

if __name__ == "__main__":
 main()