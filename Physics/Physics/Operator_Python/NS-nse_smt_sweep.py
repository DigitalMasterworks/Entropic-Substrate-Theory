from z3 import *
import csv, time


nus = [0.01, 0.1, 1.0]
Cs = [0.1, 1.0, 10.0, 100.0]
Om0s = [0.1, 1.0, 10.0]
E0s = [1.0]
dt = 0.1
N = 3
BLOWUP = 1e6

results = []
start = time.time()

for nu_val in nus:
 for C_val in Cs:
 for Om0_val in Om0s:
 for E0_val in E0s:

 s = Solver()


 E = [Real(f"E_{i}") for i in range(N+1)]
 Om = [Real(f"Om_{i}") for i in range(N+1)]


 s.add(E[0] == E0_val)
 s.add(Om[0] == Om0_val)

 for i in range(N):

 s.add(E[i+1] == E[i] - nu_val*Om[i]*dt)
 s.add(E[i+1] >= 0)



 diss_term = nu_val * (Om[i]**2) / (E[i] + 1e-9)


 s.add(Om[i+1] <= Om[i] + (C_val*Om[i]**(3/2) - diss_term)*dt)
 s.add(Om[i+1] >= 0)


 s.push()
 s.add(Or([Om[i] > BLOWUP for i in range(N+1)]))
 s.set("timeout", 5000)
 res = s.check()
 s.pop()

 results.append((nu_val,C_val,Om0_val,E0_val,res))
 print(f"nu={nu_val}, C={C_val}, Om0={Om0_val} -> {res}")

elapsed = time.time()-start
print(f"Total sweep time: {elapsed:.2f}s")


counts = {"unsat":0, "sat":0, "unknown":0}
for _,_,_,_,res in results:
 counts[str(res)] += 1
print("Summary:", counts)


with open("nse_smt_sweep_results.csv","w",newline="") as f:
 writer=csv.writer(f)
 writer.writerow(["nu","C","Om0","E0","result"])
 for row in results:
 writer.writerow(row)