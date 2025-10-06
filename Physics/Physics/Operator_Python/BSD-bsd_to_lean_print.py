#!/usr/bin/env python3


from typing import Dict, List, Tuple
import sys

ap_map = {
 2:-2,3:-3,5:-2,7:-4,11:2,13:-5,17:4,19:-2,23:1,29:-7,31:-3,37:2,41:-9,43:-8,47:9,53:2,
 59:0,61:-2,67:14,71:-3,73:-3,79:-6,83:8,89:12,97:0,101:2,103:-8,107:2,109:0,113:-20,
 127:-17,131:-15,137:-12,139:-1,149:14,151:13,157:0,163:-5,167:-8,173:-18,179:-25,181:18,
 191:2,193:17,197:9,199:-10,211:4,223:-16,227:0,229:-4,233:27,239:-3,241:-2,251:8,257:7,
 263:-10,269:13,271:8,277:-11,281:12,283:-26,293:16,307:8,311:5,313:16,317:2,331:11,337:4,
 347:-4,349:-19,353:-31,359:6,367:24,373:-34,379:-32,383:-36,389:24,397:-37,401:-6,409:-11,
 419:24,421:-20,431:24,433:34,439:-13,443:29,449:-34,457:-40,461:21,463:-8,467:22,479:16,
 487:31,491:-9,499:37,503:-6,509:-3,521:-28,523:-16,541:-15,547:-21,557:28,563:4,569:26,
 571:-4,577:33,587:-19,593:30,599:-8,601:13,607:-28,613:20,617:-30,619:-8,631:44,641:-4,
 643:10,647:17,653:3,659:-48,661:-18,673:43,677:42,683:-9,691:44,701:-12,709:-40,719:24,
 727:-30,733:28,739:-17,743:48,751:24,757:-42,761:-51,769:-26,773:-52,787:-36,797:-10,
 809:-30,811:5,821:18,823:45,827:-48,829:-30,839:-24,853:-26,857:3,859:13,863:7,877:6,
 881:36,883:12,887:-39,907:-38,911:-46,919:-16,929:15,937:-38,941:24,947:-3,953:-50,967:-13,
 971:-14,977:18,983:6,991:16,997:22
}


primes = sorted(ap_map.keys())

print("-- BSDBridge_data_generated.lean (printed to stdout)")
print("namespace Myproject")
print("namespace BSDReverse\n")
print("-- finite a_p table (paste-produced from Python run)")
print("def ap_table: List (Nat × Int):=")
print(" [")
for p in primes:
 print(f" ({p}, {ap_map[p]}),")
print(" ]\n")
print("def a_p (p: Nat): Int:=")
print(" match (ap_table.find? (fun pr => pr.1 = p)) with")
print(" | some (_, ap) => ap")
print(" | none => 0")
print(" end\n")
for p in primes:
 print(f"theorem a_p_{p}: a_p {p} = {ap_map[p]}:= by rfl")
print("\n/-- Reduction skeleton: show how the global identification reduces to finite checks + analytic remainder. -/")
print("theorem L_equals_det_reduction (P0: Nat) (finite_checks: ∀ p ∈ (ap_table.map Prod.fst).toFinset, -- local factor match predicate here )")
print(" (analytic_remainder: True): True:=")
print(" -- Replace True/placeholder above with formal analytic lemmas and the explicit local_factor_match predicate.")
print(" by trivial")
print("\nend BSDReverse")
print("end Myproject")