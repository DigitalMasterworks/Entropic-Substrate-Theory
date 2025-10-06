import numpy as np

def substrate_barrier(n, alpha=1.0):

 verify = n**2

 search = 2**n / (alpha+n)
 ratio = search/verify
 return ratio

for n in [4,8,12,16,20]:
 print(n, substrate_barrier(n))