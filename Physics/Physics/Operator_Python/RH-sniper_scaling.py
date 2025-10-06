import numpy as np
import matplotlib.pyplot as plt


lam_raw = np.load("eigs_1d/eigs_merged.npy")
lam_raw = np.sort(lam_raw[lam_raw > 0])

def fit_B(lam, c1=1.5, c2=6.0, n=40):
 lam_max = lam.max()
 t = np.geomspace(c1/lam_max, c2/lam_max, n)
 T = np.array([np.exp(-ti * lam).sum() for ti in t])
 X = np.column_stack([1/t, np.log(1/t), np.ones_like(t), t])
 coef, *_ = np.linalg.lstsq(X, T, rcond=None)
 return coef[1]


L0 = 100.0
L_list = [50, 75, 100, 125, 150, 200]
Bs = []
for L in L_list:
 scale = (L0 / L)**2
 lam = lam_raw * scale
 Bs.append(fit_B(lam))
Bs = np.array(Bs)
L_arr = np.array(L_list, float)


ratios = Bs / (L_arr**2)
print("B/L^2 values:", ratios)
print(f"mean(B/L^2) = {np.mean(ratios):.3f}, std = {np.std(ratios):.3f}")


plt.figure(figsize=(6,4))
plt.loglog(L_arr, Bs, "o-", label="fit B")
ref = Bs[0] * (L_arr/L_arr[0])**2
plt.loglog(L_arr, ref, "--", label="slope 2 ref")
plt.xlabel("Domain half-width L")
plt.ylabel("Raw B coefficient")
plt.legend()
plt.tight_layout()
plt.savefig("B_vs_L_loglog.png", dpi=150)

plt.figure(figsize=(6,4))
plt.plot(L_arr, Bs / (L_arr**2), "o-")
plt.axhline(0.25, color="r", linestyle="--", label="theory 1/4")
plt.xlabel("Domain half-width L")
plt.ylabel("B / L^2")
plt.legend()
plt.tight_layout()
plt.savefig("B_over_L2_vs_L.png", dpi=150)

print("Saved: B_vs_L_loglog.png and B_over_L2_vs_L.png")
print(f"At largest L={L_arr[-1]}, B/L^2 = {ratios[-1]:.3f}")