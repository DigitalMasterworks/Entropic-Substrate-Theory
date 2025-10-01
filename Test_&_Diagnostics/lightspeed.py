# Empirical regression fit for c = M*S (NO defined c_vac used at all)

anchor_materials = [
    # ("Material",     n,         measured c in material [m/s])
    ("Benzene",      1.501070,   199_761_471.67),
    ("Glycerol",     1.473000,   203_671_664.77),
    ("Fused Silica", 1.458500,   205_573_446.07),
    ("Ethanol",      1.361370,   220_250_303.31),
    ("Water",        1.333000,   224_906_863.53),
]

anchors = []
for _, n, c in anchor_materials:
    S = 1.0 / n
    anchors.append((S, c))

import numpy as np
S_arr, c_arr = zip(*anchors)
S_arr = np.array(S_arr)
c_arr = np.array(c_arr)

# Regression for c = M*S, through the origin
M_fit = np.sum(S_arr * c_arr) / np.sum(S_arr**2)
print(f"Derived maximal information speed (M): {M_fit:.10f} m/s")

# Extrapolate to S=1
print(f"Extrapolated 'vacuum' speed at S=1: {M_fit:.10f} m/s")