"""HIRAM kernel independent audit — 2026-09-08.
Usage: python hiram-kernel-audit-tests.py /path/to/hiram-kernel
Requires Python 3.9+ and GCC on PATH. No third-party Python packages.
Reviewed source commit: 38bc989345afc46177cca92ec4ae5cbbb52b76ae.
This harness is specific to the checked-in 12-variable/four-rule model.
If the model changes, update the independent ps/rule oracle deliberately.

Runs the existing seeded 1,000-case differential suite with a portable build
adapter, displays targeted malformed-input and scope probes, then verifies
all 3**12 evidence configurations against independent weighted truth tables.
It does not change repository sources; binaries are temporary.
Exit 1 means the exhaustive oracle found a mass/posterior/decision mismatch.
Targeted probes are diagnostic prints, not an exhaustive pass/fail contract.
A test pass would establish bounded model agreement, not physical validity,
full certification, hardware timing, or freedom from arbitrary-input overflow.
"""
import sys,ctypes,random,itertools,subprocess
from fractions import Fraction
import argparse, tempfile, pathlib, os, shutil
parser=argparse.ArgumentParser(description="Independent regression audit for the 12-variable HIRAM model reviewed at commit 38bc989.")
parser.add_argument("repo",type=pathlib.Path,help="Path to hiram-kernel checkout")
args=parser.parse_args()
repo=args.repo.resolve()
# Build into a workspace-relative tmp_test_ directory. Windows Smart App
# Control blocks execution of binaries under %LOCALAPPDATA%\Temp with
# WinError 4551, so a system temp dir makes this harness unrunnable there.
work=repo/"tmp_test_audit"
work.mkdir(parents=True,exist_ok=True)
so=work/("hiram.dll" if os.name=="nt" else "hiram.so")
cmd=["gcc","-O3","-shared"] + ([] if os.name=="nt" else ["-fPIC"])
subprocess.run(cmd+["-I"+str(repo/"include"),"-I"+str(repo/"include/hiram"),str(repo/"src/hiram_eval.c"),"-o",str(so)],check=True)
sys.path[:0]=[str(repo/"tests"),str(repo/"tools")]
import test_c_binding as t
lib=t.load_hiram_dll(str(so))
# test_c_binding.Rational dropped its to_fraction() helper in the rewrite.
def _frac(r):return Fraction(r.num,r.den)
ctx=t.HiramContext()
lib.hiram_context_init(ctypes.byref(ctx))
t.compile_shared_library=lambda *a:None
t.DLL_FILE=str(so)
random.seed(20260908)
t.run_differential_suite(1000)
print('\nTARGETED PROBES')
for name,ev,thr in [('hazard_false',{5:False},(1,20)),('hazard_false_1pct',{5:False},(1,100)),('hazard_false_high_aoa',{5:False,4:True},(1,20)),('icing',{6:True,7:True},(1,20)),('structure',{9:True,10:True},(1,20)),('no_sensors',{},(1,20)),('zero_denominator',{3:True,4:True},(1,0)),('negative_denominator',{3:True,4:True},(-1,-20))]:
 e=t.Evidence(sum(1<<v for v in ev),sum(1<<v for v,val in ev.items() if val))
 r=lib.hiram_audit_hazard(ctypes.byref(e),t.Rational(*thr),ctypes.byref(ctx))
 print(name,'decision',r.decision,'posterior',_frac(r.hazard_probability))
from hiram_compiler import compile_circuit,Clause
from hiram_ac import eval_node
for name,variables,priors,clauses in [('omitted_variable',[],{},[Clause.from_or([(0,False)])]),('empty_clause',[],{},[Clause([])]),('invalid_prior',[0],{0:Fraction(2)},[])]:
 try:
  c,r=compile_circuit(variables,priors,clauses)
  print(name,'accepted','mass',eval_node(c,r,{0:True}) if r else 0)
 except (ValueError,KeyError,TypeError) as exc:
  print(name,'rejected',type(exc).__name__,exc)
# Independent oracle: product weights and four logical implications; no compiler/evaluator reuse.
ps=[(99,100),(95,100),(1,2),(1,20),(1,10),(1,100),(1,5),(1,3),(1,50),(1,50),(1,4),(1,500)]
den=1
for a,b in ps:den*=b
N=3**12
mass=[0]*N
mismatches=audits=decisions=0
first=[]
for code in range(N):
 q=code;digits=[];om=vm=0;unknown=None;p=1
 for v in range(12):
  d=q%3;q//=3;digits.append(d)
  if d==2:
   if unknown is None:unknown=p
  else:
   om|=1<<v;vm|=d<<v
  p*=3
 if unknown is not None:mass[code]=mass[code-2*unknown]+mass[code-unknown]
 elif all(not(digits[a] and digits[b]) or digits[c] for a,b,c in [(0,1,2),(3,4,5),(6,7,8),(9,10,11)]):
  w=1
  for d,(a,b) in zip(digits,ps):w*=a if d else b-a
  mass[code]=w
 e=t.Evidence(om,vm)
 got=lib.hiram_eval_evidence(ctypes.byref(e),ctypes.byref(ctx))
 if got.num*den != mass[code]*got.den:mismatches+=1
 if mass[code]:
  # H AND E must be empty when H already observed false.
  num=0 if digits[5]==0 else mass[code-(digits[5]-1)*3**5]
  expected=Fraction(num,mass[code])
  r=lib.hiram_audit_hazard(ctypes.byref(e),t.Rational(1,20),ctypes.byref(ctx))
  if r.decision!=(1 if expected>Fraction(1,20) else 0):decisions+=1
  if _frac(r.hazard_probability)!=expected or r.decision!=(1 if expected>Fraction(1,20) else 0):
   audits+=1
   if len(first)<3:first.append((code,str(expected),str(_frac(r.hazard_probability)),r.decision))
 else:
  r=lib.hiram_audit_hazard(ctypes.byref(e),t.Rational(1,20),ctypes.byref(ctx))
  if r.decision!=2:audits+=1;decisions+=1
print('EXHAUSTIVE',N,'evidence_mass_mismatches',mismatches,'audit_mismatches',audits,'decision_mismatches_at_5pct',decisions,'first',first,flush=True)

# leave tmp_test_audit/ in place: it is gitignored, and the loaded DLL
# cannot be unlinked on Windows while the process still holds it.
sys.exit(1 if mismatches or audits else 0)
