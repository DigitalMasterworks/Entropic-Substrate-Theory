from z3 import *
from z3 import Sqrt


nu = Real('nu')
C = Real('C')
t = Real('t')
Om0 = Real('Om0')
Om = Real('Om')

s = Solver()
s.add(nu > 0, C > 0, Om0 > 0, t > 0)


denom = 1 - C * Sqrt(Om0) * t
s.add(Om <= Om0 / (denom*denom))


s.add(denom == 0)

print(s.check())
if s.check() == sat:
 m = s.model()
 print("Finite-time blow-up is consistent:")
 print("Om0 =", m[Om0], "C =", m[C], "t =", m[t])
else:
 print("No finite-time blow-up under these constraints.")