#!/usr/bin/env python3


from fractions import Fraction
import numpy as np
import sys

def rationalize(x, max_den=10**12):
 return Fraction(x).limit_denominator(max_den)

def print_lean_matrix(H, name="H_mat", max_den=10**12):

 n, m = H.shape
 print(f"-- Lean matrix {name}: size {n}x{m}")
 print(f"def {name}: Matrix (Fin {n}) (Fin {m}) Rat:=")
 print(" Array.mk " + "[")
 for i in range(n):
 row_entries = []
 for j in range(m):
 r = rationalize(float(H[i,j]), max_den=max_den)
 row_entries.append(f"{r.numerator} / {r.denominator}")
 print(" [" + ", ".join(row_entries) + "],")
 print(" ]")


if __name__ == '__main__':

 H = np.array([[1.0, -0.5], [-0.5, 1.0]])
 print_lean_matrix(H, "H_example", max_den=10**9)