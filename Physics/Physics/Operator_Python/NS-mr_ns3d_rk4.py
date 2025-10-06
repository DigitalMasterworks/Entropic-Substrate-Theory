#!/usr/bin/env python3




import argparse, csv, os
import numpy as np



def make_wave_numbers(N):
 k = np.fft.fftfreq(N) * N
 return np.meshgrid(k, k, k, indexing="ij")

def helmholtz_project(vhat, KX, KY, KZ, K2):
 """Project a velocity field (in Fourier) to divergence-free."""
 div = KX * vhat[0] + KY * vhat[1] + KZ * vhat[2]
 vhat[0] -= div * KX / K2
 vhat[1] -= div * KY / K2
 vhat[2] -= div * KZ / K2
 return vhat

def dealias_mask(N, KX, KY, KZ):
 kcut = N // 3
 return (np.abs(KX) <= kcut) & (np.abs(KY) <= kcut) & (np.abs(KZ) <= kcut)

def apply_mask(vhat, mask):
 vhat *= mask[None,:,:,:]
 return vhat

def energy(v):
 return 0.5 * np.mean(np.sum(v * v, axis=0))

def enstrophy(v, KX, KY, KZ):
 vxh = np.fft.fftn(v[0]); vyh = np.fft.fftn(v[1]); vzh = np.fft.fftn(v[2])
 dvx_dy = np.fft.ifftn(1j * KY * vxh).real; dvx_dz = np.fft.ifftn(1j * KZ * vxh).real
 dvy_dx = np.fft.ifftn(1j * KX * vyh).real; dvy_dz = np.fft.ifftn(1j * KZ * vyh).real
 dvz_dx = np.fft.ifftn(1j * KX * vzh).real; dvz_dy = np.fft.ifftn(1j * KY * vzh).real
 wx = dvy_dz - dvz_dy; wy = dvz_dx - dvx_dz; wz = dvx_dy - dvy_dx
 return 0.5 * np.mean(wx * wx + wy * wy + wz * wz)

def divergence_L2(v, KX, KY, KZ):
 vh0 = np.fft.fftn(v[0]); vh1 = np.fft.fftn(v[1]); vh2 = np.fft.fftn(v[2])
 div_hat = 1j * (KX * vh0 + KY * vh1 + KZ * vh2)
 div = np.fft.ifftn(div_hat).real
 return float(np.sqrt(np.mean(div * div)))

def energy_spectrum(v):
 """Return isotropic shell-averaged spectrum E(k)."""
 N = v.shape[1]
 k = np.fft.fftfreq(N) * N
 KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
 kk = np.sqrt(KX**2 + KY**2 + KZ**2)
 bins = np.arange(0, N//2 + 1)
 vh0 = np.fft.fftn(v[0]); vh1 = np.fft.fftn(v[1]); vh2 = np.fft.fftn(v[2])
 Ek_grid = 0.5 * (np.abs(vh0)**2 + np.abs(vh1)**2 + np.abs(vh2)**2) / (N**3)

 shell = np.rint(kk).astype(int)
 spec = np.zeros_like(bins, dtype=float)
 counts = np.zeros_like(bins, dtype=int)
 for j in range(Ek_grid.size):
 idx = shell.flatten()[j]
 if 0 <= idx < bins.size:
 spec[idx] += Ek_grid.flatten()[j].real
 counts[idx] += 1
 counts[counts == 0] = 1
 return bins, spec / counts



def init_taylor_green(N):
 x = np.linspace(0, 2*np.pi, N, endpoint=False)
 X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
 v = np.zeros((3, N, N, N), dtype=np.float64)
 v[0] = np.sin(X) * np.cos(Y) * np.cos(Z)
 v[1] = -np.cos(X) * np.sin(Y) * np.cos(Z)
 v[2] = 0.0
 return v

def init_random(N, seed=0):
 rng = np.random.default_rng(seed)
 k = np.fft.fftfreq(N) * N
 KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
 K2 = KX**2 + KY**2 + KZ**2
 K2s = np.where(K2 == 0, 1.0, K2)
 vhat = rng.normal(size=(3, N, N, N)) + 1j * rng.normal(size=(3, N, N, N))

 for i in range(3):
 vhat[i][0,:,:] = 0
 vhat[i][:, 0,:] = 0
 vhat[i][:,:, 0] = 0

 div = KX * vhat[0] + KY * vhat[1] + KZ * vhat[2]
 vhat[0] -= div * KX / K2s
 vhat[1] -= div * KY / K2s
 vhat[2] -= div * KZ / K2s
 mask = dealias_mask(N, KX, KY, KZ)
 vhat *= mask[None,:,:,:]
 return np.fft.ifftn(vhat, axes=(1, 2, 3)).real



def nonlinear_rhs(vhat, nu, KX, KY, KZ, K2, mask):
 """RHS in Fourier: -P(u·∇u) - nu*k^2*vhat"""

 v = np.fft.ifftn(vhat, axes=(1, 2, 3)).real
 ux, uy, uz = v

 uhx = np.fft.fftn(ux); uhy = np.fft.fftn(uy); uhz = np.fft.fftn(uz)
 dux_dx = np.fft.ifftn(1j*KX*uhx).real; dux_dy = np.fft.ifftn(1j*KY*uhx).real; dux_dz = np.fft.ifftn(1j*KZ*uhx).real
 duy_dx = np.fft.ifftn(1j*KX*uhy).real; duy_dy = np.fft.ifftn(1j*KY*uhy).real; duy_dz = np.fft.ifftn(1j*KZ*uhy).real
 duz_dx = np.fft.ifftn(1j*KX*uhz).real; duz_dy = np.fft.ifftn(1j*KY*uhz).real; duz_dz = np.fft.ifftn(1j*KZ*uhz).real

 conv = np.zeros_like(v)
 conv[0] = ux*dux_dx + uy*dux_dy + uz*dux_dz
 conv[1] = ux*duy_dx + uy*duy_dy + uz*duy_dz
 conv[2] = ux*duz_dx + uy*duz_dy + uz*duz_dz
 convhat = np.fft.fftn(conv, axes=(1, 2, 3))
 convhat *= mask[None,:,:,:]

 rhs = -convhat - nu * K2[None,:,:,:] * vhat
 div = KX * rhs[0] + KY * rhs[1] + KZ * rhs[2]
 rhs[0] -= div * KX / K2
 rhs[1] -= div * KY / K2
 rhs[2] -= div * KZ / K2
 return rhs

def rk4_step(vhat, dt, rhs, KX, KY, KZ, K2, mask, nu):
 k1 = rhs(vhat, nu, KX, KY, KZ, K2, mask)
 k2 = rhs(vhat + 0.5*dt*k1, nu, KX, KY, KZ, K2, mask)
 k3 = rhs(vhat + 0.5*dt*k2, nu, KX, KY, KZ, K2, mask)
 k4 = rhs(vhat + dt*k3, nu, KX, KY, KZ, K2, mask)
 vhat_new = vhat + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
 return vhat_new



def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--N", type=int, default=64)
 ap.add_argument("--nu", type=float, default=0.02)
 ap.add_argument("--dt", type=float, default=1e-3)
 ap.add_argument("--T", type=float, default=2.0)
 ap.add_argument("--ic", type=str, default="taylor-green", choices=["taylor-green","random"])
 ap.add_argument("--seed", type=int, default=0)
 ap.add_argument("--log", type=str, default="ns_log.csv")
 ap.add_argument("--spectra-every", type=int, default=0, help="dump energy spectrum every k steps (0=off)")
 ap.add_argument("--spectra-prefix", type=str, default="spec_", help="prefix for spectrum CSV files")
 ap.add_argument("--checkpoint-every", type=int, default=0, help="save v.npy every k steps (0=off)")
 ap.add_argument("--checkpoint-prefix", type=str, default="chk_")
 ap.add_argument("--cfl-max", type=float, default=0.4, help="warn/adapt if CFL exceeds this")
 ap.add_argument("--adapt", action="store_true", help="enable simple adaptive dt when CFL exceeds --cfl-max")
 args = ap.parse_args()

 N, nu, dt, Ttot = args.N, args.nu, args.dt, args.T


 if args.ic == "taylor-green":
 v = init_taylor_green(N)
 else:
 v = init_random(N, args.seed)


 E0 = np.sum(np.abs(v)**2) / (N**3)
 if E0 > 0:
 v *= (1.0 / np.sqrt(E0))
 print(f"[NS-3D] normalized E(0)=1 (was {E0:.6g})")


 KX, KY, KZ = make_wave_numbers(N)
 K2 = KX**2 + KY**2 + KZ**2
 K2[K2 == 0] = 1.0
 mask = dealias_mask(N, KX, KY, KZ)
 kmax = N // 2


 vhat = np.fft.fftn(v, axes=(1, 2, 3))
 vhat = helmholtz_project(vhat, KX, KY, KZ, K2)
 vhat = apply_mask(vhat, mask)


 os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
 with open(args.log, "w", newline="") as f:
 w = csv.writer(f)
 w.writerow(["t", "E", "Omega", "div2", "CFL", "dt"])

 steps = int(np.ceil(Ttot / dt))
 t = 0.0
 for n in range(steps):

 v_real = np.fft.ifftn(vhat, axes=(1, 2, 3)).real
 u_max = float(np.max(np.sqrt(np.sum(v_real*v_real, axis=0))))
 CFL = dt * u_max * kmax


 if args.adapt and CFL > args.cfl-max:

 dt_new = (0.8 * args.cfl_max) / (u_max * kmax + 1e-15)
 print(f"[adapt] CFL={CFL:.3f} > {args.cfl_max:.3f} -> dt {dt:.3e} -> {dt_new:.3e}")
 dt = max(dt_new, 1e-6)

 steps = n + int(np.ceil((Ttot - t) / dt)) + 1

 if CFL > args.cfl_max and not args.adapt:
 print(f"[warn] CFL={CFL:.3f} > {args.cfl_max:.3f}; consider smaller dt or enable --adapt")


 vhat = rk4_step(vhat, dt, nonlinear_rhs, KX, KY, KZ, K2, mask, nu)

 vhat = helmholtz_project(vhat, KX, KY, KZ, K2)
 vhat = apply_mask(vhat, mask)

 t += dt
 v_real = np.fft.ifftn(vhat, axes=(1, 2, 3)).real
 E = energy(v_real)
 Om = enstrophy(v_real, KX, KY, KZ)
 div2 = divergence_L2(v_real, KX, KY, KZ)
 w.writerow([f"{t:.9f}", f"{E:.9g}", f"{Om:.9g}", f"{div2:.3e}", f"{CFL:.4f}", f"{dt:.3e}"])


 if (n + 1) % max(1, steps // 10) == 0:
 print(f"[NS-3D] t={t:.3f} E={E:.6f} Ω={Om:.6f} ||div||₂={div2:.2e} CFL={CFL:.3f} dt={dt:.3e}")


 if args.spectra_every > 0 and (n + 1) % args.spectra_every == 0:
 bins, spec = energy_spectrum(v_real)
 spath = f"{args.spectra_prefix}{n+1:06d}.csv"
 with open(spath, "w", newline="") as sf:
 sw = csv.writer(sf)
 sw.writerow(["k", "E(k)"])
 for kk, ee in zip(bins, spec):
 sw.writerow([kk, f"{ee:.9e}"])


 if args.checkpoint_every > 0 and (n + 1) % args.checkpoint_every == 0:
 np.save(f"{args.checkpoint_prefix}{n+1:06d}.npy", v_real.astype(np.float32))

 if t >= Ttot - 1e-15:
 break

 print(f"[NS-3D] done. log -> {args.log}")

if __name__ == "__main__":
 main()