#!/usr/bin/env python3

import numpy as np
from itertools import product
from math import pi, sqrt






V = 4*pi*pi

b0 = lambda x,y: 1.0 / sqrt(V)
b1x = lambda x,y: 1.0/sqrt(V)
b1y = lambda x,y: 1.0/sqrt(V)
b2 = lambda x,y: 1.0/sqrt(V)










P_traces = {
 'P0_trace': 1.0,
 'P1_trace': 2.0,
 'P2_trace': 1.0
}

print("Analytic torus zero-mode projector traces (expected):", P_traces)



R = 60
modes = []
for nx in range(-R,R+1):
 for ny in range(-R,R+1):
 k2 = nx*nx + ny*ny
 modes.append(((nx,ny), k2))
modes = sorted(modes, key=lambda t: (t[1], t[0]))

zero_modes = [m for m in modes if m[1]==0]
print("zero_modes count (discrete):", len(zero_modes))


print("Discrete Fourier truncation reports zero eigenvalue multiplicity:", len(zero_modes))