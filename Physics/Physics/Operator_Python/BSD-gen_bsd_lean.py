#!/usr/bin/env python3
"""
gen_bsd_lean.py

Reads the JSON file (from full_bridge_tools.py) and prints Lean theorem lines
like:

 theorem a_p_2: a_p 2 = -2:= by rfl

so you never have to hand-type them again

Usage:
 python3 gen_bsd_lean.py bsd_ap_table.json > BSD_theorems.lean
"""

import json
import sys

def main():
 if len(sys.argv) < 2:
 print("Usage: python3 gen_bsd_lean.py bsd_ap_table.json")
 sys.exit(1)

 with open(sys.argv[1]) as f:
 data = json.load(f)

 ap_table = data.get("ap_table", {})


 for p_str, ap in sorted(ap_table.items(), key=lambda kv: int(kv[0])):
 p = int(p_str)
 print(f"theorem a_p_{p}: a_p {p} = {ap}:= by rfl")

if __name__ == "__main__":
 main()