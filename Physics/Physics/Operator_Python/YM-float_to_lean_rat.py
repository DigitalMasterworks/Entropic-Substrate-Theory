
from fractions import Fraction
def to_lean_rat(x, max_den=10**12):
 f = Fraction(x).limit_denominator(max_den)
 return f"{f.numerator} / {f.denominator}"

print(to_lean_rat(0.5857864376269043))