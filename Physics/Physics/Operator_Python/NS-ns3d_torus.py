import argparse
import numpy as np
import matplotlib.pyplot as plt
def init_taylor_green(N):
 """
 Taylor–Green vortex initial condition on [0,2π]^3
 v = (sin x cos y cos z, -cos x sin y cos z, 0)
 """
 x = np.linspace(0, 2*np.pi, N, endpoint=False)
 y = np.linspace(0, 2*np.pi, N, endpoint=False)
 z = np.linspace(0, 2*np.pi, N, endpoint=False)
 X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
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
 K2_safe = np.where(K2 == 0, 1.0, K2)
 vhat = rng.normal(size=(3,N,N,N)) + 1j*rng.normal(size=(3,N,N,N))
 for i in range(3):
 vhat[i][0,:,:] = 0
 vhat[i][:,0,:] = 0
 vhat[i][:,:,0] = 0
 div = KX*vhat[0] + KY*vhat[1] + KZ*vhat[2]
 vhat[0] -= div*KX/K2_safe
 vhat[1] -= div*KY/K2_safe
 vhat[2] -= div*KZ/K2_safe
 kcut = N//3
 mask = (np.abs(KX) <= kcut) & (np.abs(KY) <= kcut) & (np.abs(KZ) <= kcut)
 vhat *= mask[None,:,:,:]
 v = np.fft.ifftn(vhat, axes=(1,2,3)).real
 return v

def ns_rhs(vhat, nu, KX, KY, KZ, K2, mask):
 """
 dv/dt = -P(u·∇u) - ν|k|^2 v
 Use spectral derivatives and a shared 2/3 mask.
 """

 v = np.fft.ifftn(vhat, axes=(1, 2, 3)).real
 ux, uy, uz = v


 uhx = np.fft.fftn(ux, axes=(0, 1, 2))
 uhy = np.fft.fftn(uy, axes=(0, 1, 2))
 uhz = np.fft.fftn(uz, axes=(0, 1, 2))

 dux_dx = np.fft.ifftn(1j * KX * uhx, axes=(0, 1, 2)).real
 dux_dy = np.fft.ifftn(1j * KY * uhx, axes=(0, 1, 2)).real
 dux_dz = np.fft.ifftn(1j * KZ * uhx, axes=(0, 1, 2)).real

 duy_dx = np.fft.ifftn(1j * KX * uhy, axes=(0, 1, 2)).real
 duy_dy = np.fft.ifftn(1j * KY * uhy, axes=(0, 1, 2)).real
 duy_dz = np.fft.ifftn(1j * KZ * uhy, axes=(0, 1, 2)).real

 duz_dx = np.fft.ifftn(1j * KX * uhz, axes=(0, 1, 2)).real
 duz_dy = np.fft.ifftn(1j * KY * uhz, axes=(0, 1, 2)).real
 duz_dz = np.fft.ifftn(1j * KZ * uhz, axes=(0, 1, 2)).real


 conv = np.zeros_like(v)
 conv[0] = ux * dux_dx + uy * dux_dy + uz * dux_dz
 conv[1] = ux * duy_dx + uy * duy_dy + uz * duy_dz
 conv[2] = ux * duz_dx + uy * duz_dy + uz * duz_dz


 convhat = np.fft.fftn(conv, axes=(1, 2, 3))
 convhat *= mask[None,:,:,:]


 rhs = -convhat - nu * K2[None,:,:,:] * vhat
 div = KX * rhs[0] + KY * rhs[1] + KZ * rhs[2]
 rhs[0] -= div * KX / K2
 rhs[1] -= div * KY / K2
 rhs[2] -= div * KZ / K2
 return rhs

def integrate_ns(v, nu, dt, T):
 """
 Simple RK2 integrator for Navier–Stokes.
 """
 N = v.shape[1]
 k = np.fft.fftfreq(N)*N
 KX, KY, KZ = np.meshgrid(k,k,k,indexing="ij")
 K2 = KX**2 + KY**2 + KZ**2
 K2[K2==0] = 1.0

 vhat = np.fft.fftn(v, axes=(1, 2, 3))


 kcut = N // 3
 mask = (np.abs(KX) <= kcut) & (np.abs(KY) <= kcut) & (np.abs(KZ) <= kcut)


 div0 = KX * vhat[0] + KY * vhat[1] + KZ * vhat[2]
 vhat[0] -= div0 * KX / K2
 vhat[1] -= div0 * KY / K2
 vhat[2] -= div0 * KZ / K2
 vhat *= mask[None,:,:,:]

 u0 = np.fft.ifftn(vhat, axes=(1, 2, 3)).real
 umax0 = float(np.max(np.sqrt(np.sum(u0*u0, axis=0))))
 kmax = N // 2
 cfl0 = dt * umax0 * kmax
 print(f"[NS-3D] initial CFL≈{cfl0:.3f} (u_max={umax0:.3f}, k_max={kmax})")
 steps = int(T/dt)
 for n in range(steps):

 divp = KX * vhat[0] + KY * vhat[1] + KZ * vhat[2]
 vhat[0] -= divp * KX / K2
 vhat[1] -= divp * KY / K2
 vhat[2] -= divp * KZ / K2
 vhat *= mask[None,:,:,:]
 k1 = ns_rhs(vhat, nu, KX, KY, KZ, K2, mask)
 k2 = ns_rhs(vhat + 0.5 * dt * k1, nu, KX, KY, KZ, K2, mask)
 vhat = vhat + dt*k2


 div = KX * vhat[0] + KY * vhat[1] + KZ * vhat[2]
 vhat[0] -= div * KX / K2
 vhat[1] -= div * KY / K2
 vhat *= mask[None,:,:,:]
 if (n + 1) % max(1, steps // 10) == 0:
 v_snap = np.fft.ifftn(vhat, axes=(1,2,3)).real
 E_snap = 0.5 * np.mean(np.sum(v_snap*v_snap, axis=0))
 vh0 = np.fft.fftn(v_snap[0], axes=(0, 1, 2))
 vh1 = np.fft.fftn(v_snap[1], axes=(0, 1, 2))
 vh2 = np.fft.fftn(v_snap[2], axes=(0, 1, 2))
 div_hat = 1j * (KX * vh0 + KY * vh1 + KZ * vh2)
 div = np.fft.ifftn(div_hat, axes=(0, 1, 2)).real
 div_l2 = float(np.sqrt(np.mean(div * div)))
 print(f"[NS-3D] t={(n+1)*dt:.4f} E={E_snap:.6f} ||div u||_2={div_l2:.2e}")
 v = np.fft.ifftn(vhat, axes=(1,2,3)).real
 return v

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--N", type=int, default=32, help="Grid size (number of modes per dimension)")
 ap.add_argument("--nu", type=float, default=0.01, help="Viscosity coefficient")
 ap.add_argument("--dt", type=float, default=1e-3, help="Time step")
 ap.add_argument("--T", type=float, default=0.5, help="Total integration time")
 ap.add_argument("--ic", type=str, default="taylor-green",
 choices=["taylor-green", "random"], help="Initial condition")
 ap.add_argument("--seed", type=int, default=0, help="Random seed for random IC")
 args = ap.parse_args()

 if args.ic == "taylor-green":
 v = init_taylor_green(args.N)
 elif args.ic == "random":
 v = init_random(args.N, args.seed)

 E0 = np.sum(np.abs(v)**2) / (args.N**3)
 if E0 > 0:
 v *= (1.0 / np.sqrt(E0))
 print(f"[NS-3D] normalized initial energy to 1.0 (was {E0:.6g})")

 k = np.fft.fftfreq(args.N) * args.N
 KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
 K2 = KX**2 + KY**2 + KZ**2
 K2[K2==0] = 1.0

 kcut = args.N // 3
 mask = (np.abs(KX) <= kcut) & (np.abs(KY) <= kcut) & (np.abs(KZ) <= kcut)

 vhat0 = np.fft.fftn(v, axes=(1, 2, 3))
 div0 = KX * vhat0[0] + KY * vhat0[1] + KZ * vhat0[2]
 vhat0[0] -= div0 * KX / K2
 vhat0[1] -= div0 * KY / K2
 vhat0[2] -= div0 * KZ / K2
 vhat0 *= mask[None,:,:,:]
 v = np.fft.ifftn(vhat0, axes=(1, 2, 3)).real

 import matplotlib.pyplot as plt
 plt.ion()
 fig, ax = plt.subplots(2, 1, figsize=(6, 8))
 t_vals, E_vals, div_vals = [], [], []
 ax[0].set_title("Energy decay")
 ax[0].set_xlabel("t")
 ax[0].set_ylabel("E(t)")
 ax[1].set_title("Divergence norm")
 ax[1].set_xlabel("t")
 ax[1].set_ylabel("||div u||₂")

 vhat = np.fft.fftn(v, axes=(1,2,3))
 steps = int(args.T/args.dt)
 for n in range(steps):
 k1 = ns_rhs(vhat, args.nu, KX, KY, KZ, K2, mask)
 k2 = ns_rhs(vhat + 0.5*args.dt*k1, args.nu, KX, KY, KZ, K2, mask)
 vhat = vhat + args.dt*k2

 div = KX*vhat[0] + KY*vhat[1] + KZ*vhat[2]
 vhat[0] -= div*KX/K2
 vhat[1] -= div*KY/K2
 vhat[2] -= div*KZ/K2
 vhat *= mask[None,:,:,:]

 if (n+1) % max(1, steps//50) == 0 or n==steps-1:
 v_snap = np.fft.ifftn(vhat, axes=(1,2,3)).real
 E_snap = 0.5*np.mean(np.sum(v_snap*v_snap, axis=0))
 vh0 = np.fft.fftn(v_snap[0], axes=(0,1,2))
 vh1 = np.fft.fftn(v_snap[1], axes=(0,1,2))
 vh2 = np.fft.fftn(v_snap[2], axes=(0,1,2))
 div_hat = 1j*(KX*vh0 + KY*vh1 + KZ*vh2)
 div_field = np.fft.ifftn(div_hat, axes=(0,1,2)).real
 div_l2 = float(np.sqrt(np.mean(div_field*div_field)))

 t = (n+1)*args.dt
 t_vals.append(t)
 E_vals.append(E_snap)
 div_vals.append(div_l2)

 ax[0].cla(); ax[0].set_title("Energy decay"); ax[0].set_xlabel("t"); ax[0].set_ylabel("E(t)")
 ax[0].plot(t_vals, E_vals, 'b-')
 ax[1].cla(); ax[1].set_title("Divergence norm"); ax[1].set_xlabel("t"); ax[1].set_ylabel("||div u||₂")
 ax[1].plot(t_vals, div_vals, 'r-')
 plt.pause(0.01)

 v = np.fft.ifftn(vhat, axes=(1,2,3)).real
 E = 0.5 * np.mean(np.sum(v**2, axis=0))

 vx, vy, vz = v
 vxh = np.fft.fftn(vx, axes=(0,1,2))
 vyh = np.fft.fftn(vy, axes=(0,1,2))
 vzh = np.fft.fftn(vz, axes=(0,1,2))
 dvx_dy = np.fft.ifftn(1j*KY*vxh, axes=(0,1,2)).real
 dvx_dz = np.fft.ifftn(1j*KZ*vxh, axes=(0,1,2)).real
 dvy_dx = np.fft.ifftn(1j*KX*vyh, axes=(0,1,2)).real
 dvy_dz = np.fft.ifftn(1j*KZ*vyh, axes=(0,1,2)).real
 dvz_dx = np.fft.ifftn(1j*KX*vzh, axes=(0,1,2)).real
 dvz_dy = np.fft.ifftn(1j*KY*vzh, axes=(0,1,2)).real
 ωx = dvy_dz - dvz_dy
 ωy = dvz_dx - dvx_dz
 ωz = dvx_dy - dvy_dx
 Ω = 0.5 * np.mean(ωx**2 + ωy**2 + ωz**2)

 print(f"[NS-3D] N={args.N}, nu={args.nu}, dt={args.dt}, T={args.T}, ic={args.ic}")
 print(f" Final energy={E:.6f}, enstrophy={Ω:.6f}")

 plt.ioff()
 plt.show()

if __name__ == "__main__":
 main()