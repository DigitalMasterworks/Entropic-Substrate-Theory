import numpy as np
from fractions import Fraction




def rat_det_topk(H, k, max_den=10**12):
 import numpy.linalg as la
 sub = H[:k,:k]

 detf = float(np.linalg.det(sub))
 return Fraction(detf).limit_denominator(max_den)

for k in range(1, 25):
 print(k, rat_det_topk(H, k))