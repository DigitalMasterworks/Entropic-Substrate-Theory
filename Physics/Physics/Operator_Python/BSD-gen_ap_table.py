#!/usr/bin/env python3
import argparse, csv, math

def sieve_primes(n):
 n = int(n)
 bs = bytearray(b"\x01")*(n+1)
 bs[0:2] = b"\x00\x00"
 for i in range(2, int(n**0.5)+1):
 if bs[i]:
 step = i
 start = i*i
 bs[start:n+1:step] = b"\x00"*(((n - start)//step)+1)
 return [i for i in range(n+1) if bs[i]]

def discriminant(A,B):

 return -16*(4*A*A*A + 27*B*B)

def legendre_symbol(a, p):
 if a % p == 0:
 return 0
 t = pow(a, (p-1)//2, p)
 if t == 1:
 return 1

 return -1

def ap_for_prime(p, A, B):

 if p == 2:
 return None, False

 Δ = discriminant(A,B)
 if Δ % p == 0:
 return None, False

 s = 0
 for x in range(p):
 rhs = (x*x % p * x + A*x + B) % p
 s += legendre_symbol(rhs, p)
 ap = -s
 return ap, True

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--A", type=int, required=True)
 ap.add_argument("--B", type=int, required=True)
 ap.add_argument("--Pmax", type=int, default=2000)
 ap.add_argument("--out", type=str, default="ap_table.csv")
 args = ap.parse_args()

 primes = sieve_primes(args.Pmax)
 rows = []
 for p in primes:
 apv, good = ap_for_prime(p, args.A, args.B)
 if good:
 rows.append((p, apv, 1))
 else:
 rows.append((p, 0, 0))

 with open(args.out, "w", newline="") as f:
 w = csv.writer(f)
 w.writerow(["p","ap","good"])
 for p, apv, good in rows:
 w.writerow([p, apv, good])
 print(f"[ap] wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
 main()