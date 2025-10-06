
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sla
import numpy as np
import math

def spacing_ratios(eigs):
 lam = np.sort(np.asarray(eigs))
 d = np.diff(lam)
 r = d[1:]/d[:-1]
 return np.minimum(r, 1.0/r)

def five_point_L(N, h):

 main = cp.full(N*N, -4.0)
 east = cp.ones(N*N-1); east[cp.arange(1,N*N)%N==0] = 0.0
 west = cp.ones(N*N-1); west[cp.arange(0,N*N-1)%N==N-1] = 0.0
 north = cp.ones(N*N-N)
 south = cp.ones(N*N-N)
 diags = [main, east, west, north, south]
 offs = [0, +1, -1, +N, -N]
 L = sp.diags(diags, offs, shape=(N*N, N*N), format="csr") / (h*h)
 return L

def five_point_L_magnetic(S, h, alpha):

 N = S.shape[0]

 logS = cp.log(S + 1e-18)
 dlogS_dx = cp.zeros_like(S); dlogS_dy = cp.zeros_like(S)
 dlogS_dx[1:-1,:] = (logS[2:,:] - logS[:-2,:])/(2*h)
 dlogS_dy[:,1:-1] = (logS[:,2:] - logS[:,:-2])/(2*h)


 A_x = alpha * dlogS_dy
 A_y = -alpha * dlogS_dx



 A_x_mid = cp.zeros_like(S); A_y_mid = cp.zeros_like(S)
 A_x_mid[:-1,:] = 0.5*(A_x[:-1,:] + A_x[1:,:])
 A_y_mid[:,:-1] = 0.5*(A_y[:,:-1] + A_y[:,1:])


 rows = []; cols = []; data = []

 def add_triplet(i, j, ii, jj, val):
 rows.append(i*N + j)
 cols.append(ii*N + jj)

 if hasattr(val, "item"):
 data.append(complex(val.item()))
 else:
 data.append(val)


 for i in range(N):
 for j in range(N):
 add_triplet(i,j,i,j,-4.0)


 for i in range(N-1):
 for j in range(N):
 phase_e = cp.exp(1j * h * A_x_mid[i,j])

 add_triplet(i, j, i+1, j, phase_e)

 phase_w = cp.conj(phase_e)
 add_triplet(i+1, j, i, j, phase_w)


 for i in range(N):
 for j in range(N-1):
 phase_n = cp.exp(1j * h * A_y_mid[i,j])
 add_triplet(i, j, i, j+1, phase_n)
 phase_s = cp.conj(phase_n)
 add_triplet(i, j+1, i, j, phase_s)

 rows = cp.asarray(rows, dtype=cp.int32)
 cols = cp.asarray(cols, dtype=cp.int32)
 data = cp.asarray(data, dtype=cp.complex128)
 L_A = sp.csr_matrix((data, (rows, cols)), shape=(N*N, N*N))
 return L_A / (h*h)

def five_point_L_periodic(N, h):
 import cupy as cp, cupyx.scipy.sparse as sp

 e = cp.ones(N)
 T = sp.diags([e, -2*e, e], [-1, 0, 1], shape=(N, N), format="csr")
 T = T.tolil()
 T[0, N-1] = 1.
 T[N-1, 0] = 1.
 T = T.tocsr()
 I = sp.identity(N, format="csr")
 L = (sp.kron(I, T) + sp.kron(T, I)) / (h*h)
 return L

def unfold(lams):

 l = np.sort(np.asarray(lams))
 idx = np.arange(1, len(l)+1)

 coefs = np.polyfit(l, idx, 3)
 Nhat = np.poly1d(coefs)
 lam_u = Nhat(l)
 s = np.diff(lam_u)
 return s

def local_unfold(lams, win=51):
 l = np.sort(np.asarray(lams))
 s_raw = np.diff(l)
 pad = win//2
 s_pad = np.pad(s_raw, (pad,pad), mode='edge')
 m = np.convolve(s_pad, np.ones(win)/win, mode='valid')
 return s_raw / m

def wigner_gue_pdf(s, ensemble="GUE"):
 s = np.asarray(s)
 if ensemble.upper() == "GUE":

 return (32.0/np.pi**2) * s**2 * np.exp(-4*s**2/np.pi)
 elif ensemble.upper() == "GOE":

 return (np.pi/2.0) * s * np.exp(-np.pi*s*s/4.0)
 else:
 raise ValueError("ensemble must be 'GOE' or 'GUE'")

def ks_to_ensemble(s, ensemble="GUE", bins=2000):
 s = np.asarray(s)
 s = s[(s>0) & np.isfinite(s)]
 s = s / np.mean(s)
 xs = np.linspace(0, 3.0, bins)
 pdf = wigner_gue_pdf(xs, ensemble=ensemble)
 cdf = np.cumsum(pdf); cdf /= cdf[-1]

 s_sorted = np.sort(s)
 import bisect
 def emp_at(x):
 j = bisect.bisect_right(s_sorted, x)
 return j/len(s_sorted)

 diffs = np.array([abs(emp_at(x)-c) for x,c in zip(xs,cdf)])
 return xs[diffs.argmax()], diffs.max()

def ks_to_gue(s, bins=2000):
 s = np.asarray(s)
 s = s[(s>0) & np.isfinite(s)]
 s = s / np.mean(s)
 xs = np.linspace(0, 3.0, bins)
 pdf = wigner_gue_pdf(xs)
 cdf = np.cumsum(pdf); cdf /= cdf[-1]

 s_sorted = np.sort(s)
 import bisect
 def emp_at(x):
 j = bisect.bisect_right(s_sorted, x)
 return j/len(s_sorted)
 diffs = np.array([abs(emp_at(x)-c) for x,c in zip(xs,cdf)])
 return xs[diffs.argmax()], diffs.max()

if __name__ == "__main__":
 import sys
 import cupy as cp

 print("CuPy available devices:", cp.cuda.runtime.getDeviceCount())
 dev = cp.cuda.Device()
 fname = sys.argv[1] if len(sys.argv)>1 else "S_halo.npy"


 S_cpu = np.load(fname)
 S = cp.array(S_cpu)

 N = S.shape[0]
 Lbox = 100.0
 h = Lbox/(N-1)



 alpha = 0.0
 if len(sys.argv) >= 3 and sys.argv[2].startswith("--mag"):
 try:
 alpha = float(sys.argv[2].split("=")[1])
 except:
 alpha = 1e-3


 if alpha == 0.0:
 L = five_point_L(N, h)
 else:
 L = five_point_L_magnetic(S, h, alpha)

 D = sp.diags((S*S).reshape(-1), 0, format="csr")
 H = -(D @ L)


 k = 600
 vals, _ = sla.eigsh(H, k=k, which="SA")
 vals = cp.asnumpy(vals)
 vals_np = np.sort(vals)
 lo, hi = int(0.10*len(vals_np)), int(0.90*len(vals_np))
 vals_bulk = vals_np[lo:hi]

 r = spacing_ratios(vals_bulk)
 print(f"bulk mean r~ = {r.mean():.5f} (GOE~0.53590, GUE~0.60266)")

 r = spacing_ratios(vals)
 print(f"mean r~ = {r.mean():.5f} (GOE~0.53590, GUE~0.60266)")


 np.savetxt(fname.replace(".npy",".eigs.txt"), np.sort(vals))
 s = unfold(vals)
 xworst, ks = ks_to_gue(s)
 print(f"{fname}: k={k} eigenvals, KS_dist_to_GUE={ks:.4f} at s≈{xworst:.2f}")
 s = unfold(vals)
 x_gue, ks_gue = ks_to_ensemble(s, "GUE")
 x_goe, ks_goe = ks_to_ensemble(s, "GOE")
 print(f"{fname}: k={k} eigenvals | KS_GUE={ks_gue:.4f} at s≈{x_gue:.2f} | KS_GOE={ks_goe:.4f} at s≈{x_goe:.2f}")
 s_loc = local_unfold(vals_bulk, win=51)
 xg, ksg = ks_to_ensemble(s_loc, "GUE")
 xo, kso = ks_to_ensemble(s_loc, "GOE")
 print(f"[local] KS_GUE={ksg:.4f} | KS_GOE={kso:.4f} | spacings={len(s_loc)}")