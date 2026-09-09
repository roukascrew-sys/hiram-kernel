"""Additional HIRAM regression tests for commit feb66bd (2026-09-08).
Windows-adapted regression suite with workspace-isolated scratchpad and file-lock defense.
"""
import argparse, ctypes as ct, itertools, json, math, pathlib, random, subprocess, sys, tempfile, os, shutil
from fractions import Fraction as F

p = argparse.ArgumentParser()
p.add_argument('repo', type=pathlib.Path)
a = p.parse_args()
repo = a.repo.resolve()

sys.path.insert(0, str(repo / 'tools'))
from hiram_ac import eval_node
from hiram_compiler import compile_circuit, Clause

rng = random.Random(20260908)

class R(ct.Structure):
    _fields_ = [('num', ct.c_int64), ('den', ct.c_int64)]

class E(ct.Structure):
    _fields_ = [('observed_mask', ct.c_uint32), ('values_mask', ct.c_uint32)]

class Report(ct.Structure):
    _fields_ = [('decision', ct.c_int), ('prob', R), ('threshold', R)]

class Ctx(ct.Structure):
    _fields_ = [('memo', R * 81), ('execution_flags', ct.c_uint32)]

result = {
    'seed': 20260908,
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
}

if sys.platform == 'win32':
    gcc_bin = shutil.which('gcc')
    if gcc_bin:
        try: os.add_dll_directory(os.path.dirname(os.path.abspath(gcc_bin)))
        except Exception: pass

tmp_dir = tempfile.mkdtemp(prefix='tmp_test_rigor_', dir=repo)
tmp = pathlib.Path(tmp_dir)
loaded_libs = []

try:
    if sys.platform == 'win32':
        try: os.add_dll_directory(str(tmp))
        except Exception: pass

    def build(name, extra, source=None):
        if sys.platform == 'win32':
            if name.endswith('.so'):
                name = name[:-3] + '.dll'
            elif not name.endswith('.exe') and not name.endswith('.dll'):
                name = name + '.exe'
        dest = tmp / name
        cmd = ['gcc', '-O2', '-g', '-I' + str(repo / 'include/hiram')] + extra
        if source:
            cmd.append(str(source))
        else:
            cmd.append(str(repo / 'src/hiram_eval.c'))
        cmd += ['-o', str(dest)]
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"\n[BUILD ERROR] Command failed: {' '.join(cmd)}")
            print(f"STDERR:\n{proc.stderr}")
            print(f"STDOUT:\n{proc.stdout}")
            sys.exit(1)
        return dest

    libs = []
    for name, flags in [('wide.so', []), ('fallback.so', ['-DHIRAM_FORCE_64BIT_ACC'])]:
        built_dest = build(name, ['-shared', '-fPIC'] + flags)
        if sys.platform == 'win32':
            lib = ct.CDLL(str(built_dest), winmode=0)
        else:
            lib = ct.CDLL(str(built_dest))
        loaded_libs.append(lib)
        lib.hiram_context_init.argtypes = [ct.POINTER(Ctx)]
        lib.hiram_audit_hazard.argtypes = [ct.POINTER(E), R, ct.POINTER(Ctx)]
        lib.hiram_audit_hazard.restype = Report
        lib.hiram_evidence_set.argtypes = [ct.POINTER(E), ct.c_uint16, ct.c_bool]
        libs.append(lib)

    # Exact comparator oracle for a fixed independent model result
    posterior = F(200, 19901)
    ev = E(0, 0)
    thresholds = [(1, 2**63 - 1), (1, 2**63 - 2), (0, 1), (1, 1), (200, 19901), (199, 19901), (201, 19901)]
    for i in range(10000):
        den = rng.randrange(1, 2**63)
        num = rng.choice([0, 1, rng.randrange(den + 1), den // 100, den // 20])
        thresholds.append((num, den))

    comparison = []
    for label, lib in zip(['host_128', 'host_forced_64'], libs):
        fails = []
        count = 0
        ctx = Ctx()
        lib.hiram_context_init(ct.byref(ctx))
        for n, d in thresholds:
            r = lib.hiram_audit_hazard(ct.byref(ev), R(n, d), ct.byref(ctx))
            expected = int(posterior > F(n, d))
            if r.decision != expected:
                count += 1
                if len(fails) < 5:
                    fails.append({'threshold': [n, d], 'expected': expected, 'actual': r.decision})
        comparison.append({'build': label, 'cases': len(thresholds), 'mismatches': count, 'examples': fails})
    result['threshold_comparison'] = comparison

    # Malformed input diagnostics
    ctx0 = Ctx()
    libs[0].hiram_context_init(ct.byref(ctx0))
    diagnostics = []
    for n, d in [(1, 0), (-1, 20), (1, -20), (-1, -20), (2, 1)]:
        r = libs[0].hiram_audit_hazard(ct.byref(ev), R(n, d), ct.byref(ctx0))
        diagnostics.append({'threshold': [n, d], 'decision': r.decision})
    for v in [12, 31, 32, 65535]:
        e = E(0, 0)
        libs[0].hiram_evidence_set(ct.byref(e), v, True)
        r = libs[0].hiram_audit_hazard(ct.byref(e), R(1, 20), ct.byref(ctx0))
        diagnostics.append({'var_id': v, 'observed_mask': e.observed_mask, 'decision': r.decision})
    result['input_diagnostics'] = diagnostics

    ctx1 = Ctx()
    libs[1].hiram_context_init(ct.byref(ctx1))
    scaled = []
    for k in [1, 10**14, 10**15, 10**16, 10**17, 461168601842738790]:
        r = libs[1].hiram_audit_hazard(ct.byref(ev), R(k, 20 * k), ct.byref(ctx1))
        scaled.append({'threshold': [k, 20 * k], 'expected': 0, 'actual': r.decision})
    result['equivalent_5pct_thresholds_forced_64'] = scaled

    # Sanity Runner: circuit data header first (defines dims, includes hiram_eval.h)
    source = tmp / 'sanity.c'
    source.write_text(r'''
#include "hiram_circuit_data.h"
#include <stdio.h>
int main(int argc,char**argv){
 HiramContext ctx;
 hiram_context_init(&ctx);
 if(argc>1){Evidence e={0,0};HiramAuditReport r=hiram_audit_hazard(&e,(Rational){1,9223372036854775806LL},&ctx);printf("decision=%d\n",r.decision);return r.decision==1?0:2;}
 unsigned count=0;
 for(unsigned code=0;code<531441;code++){
  Evidence e={0,0};unsigned q=code;
  for(unsigned v=0;v<12;v++){unsigned d=q%3;q/=3;if(d!=2)hiram_evidence_set(&e,v,d!=0);}
  HiramAuditReport r=hiram_audit_hazard(&e,(Rational){1,20},&ctx);
  if(r.hazard_probability.den<=0||r.hazard_probability.num<0||r.hazard_probability.num>r.hazard_probability.den)return 3;
  if(r.decision!=2){
   HiramAuditReport eq=hiram_audit_hazard(&e,r.hazard_probability,&ctx);
   if(eq.decision!=0)return 4;
   if(r.hazard_probability.num>0){
    HiramAuditReport lo=hiram_audit_hazard(&e,(Rational){r.hazard_probability.num-1,r.hazard_probability.den},&ctx);
    if(lo.decision!=1)return 5;
   }
   if(r.hazard_probability.num<r.hazard_probability.den){
    HiramAuditReport hi=hiram_audit_hazard(&e,(Rational){r.hazard_probability.num+1,r.hazard_probability.den},&ctx);
    if(hi.decision!=0)return 6;
   }
  }
  count++;
 }
 printf("evidence_states=%u; threshold equality and neighboring fractions passed\n",count);return 0;
}''')
    
    ubsan_flags = []
    if sys.platform != 'win32':
        probe_proc = subprocess.run(['gcc', '-fsanitize=undefined', '-x', 'c', '-', '-o', '/dev/null'], input='int main(){return 0;}', text=True, capture_output=True)
        if probe_proc.returncode == 0:
            ubsan_flags = ['-fsanitize=undefined', '-fno-sanitize-recover=undefined']

    binary = build('sanity', ['-DHIRAM_FORCE_64BIT_ACC'] + ubsan_flags + [str(repo / 'src/hiram_eval.c')], source)
    san = []
    for args in [[], ['extreme']]:
        proc = subprocess.run([str(binary)] + args, capture_output=True, text=True, timeout=120)
        san.append({
            'case': 'extreme_threshold' if args else 'exhaustive_fallback_boundaries',
            'exit': proc.returncode,
            'stdout': proc.stdout.strip(),
            'stderr': proc.stderr.strip()
        })
    result['sanitizer'] = san

    # Regeneration check
    gen = tmp / 'generated'
    gen.mkdir()
    subprocess.run([sys.executable, str(repo / 'tools/hiram_c_exporter.py')], cwd=gen, check=True, capture_output=True, text=True)
    result['regeneration'] = {
        name: (gen / name).read_bytes() == (repo / dest).read_bytes()
        # Only the generated artifact. The other two are hand-written sources;
        # diffing a copy of them against themselves proved nothing.
        for name, dest in [
            ('hiram_circuit_data.h', 'include/hiram/hiram_circuit_data.h')
        ]
    }

    # Independent random-CNF oracle
    checks = bad = 0
    examples = []
    for model in range(1000):
        n = rng.randrange(1, 7)
        vs = list(range(n))
        rng.shuffle(vs)
        priors = {v: F(rng.randrange(11), 10) for v in vs}
        raw = [[(rng.randrange(n), bool(rng.randrange(2))) for _ in range(rng.randrange(5))] for _ in range(rng.randrange(9))]
        clauses = [Clause.from_or(ls) for ls in raw]
        c, root = compile_circuit(vs, priors, clauses)
        worlds = []
        for bits in itertools.product([False, True], repeat=n):
            if all(any(bits[v] != neg for v, neg in ls) for ls in raw):
                w = F(1)
                for v in range(n):
                    w *= priors[v] if bits[v] else 1 - priors[v]
                worlds.append((bits, w))
        evs = [dict(enumerate(bits)) for bits in itertools.product([False, True], repeat=n)]
        evs += [{v: bool(d) for v in range(n) if (d := rng.randrange(3)) != 2} for _ in range(20)]
        evs.append({})
        for evidence in evs:
            expected = sum((w for bits, w in worlds if all(bits[v] == val for v, val in evidence.items())), F(0))
            got = F(0) if root is None else eval_node(c, root, evidence)
            checks += 1
            if got != expected:
                bad += 1
                if len(examples) < 3:
                    examples.append({'model': model, 'expected': str(expected), 'actual': str(got)})
    result['random_cnf'] = {'models': 1000, 'checks': checks, 'mismatches': bad, 'examples': examples}

    # Compiler invalid input check
    invalid = []
    for name, vs, priors, clauses in [
        ('empty_contradictory_clause', [], {}, [Clause([])]),
        ('omitted_variable', [], {}, [Clause.from_or([(0, False)])]),
        ('prior_above_one', [0], {0: F(2)}, [])
    ]:
        try:
            c, r = compile_circuit(vs, priors, clauses)
            invalid.append({'case': name, 'accepted': True, 'mass': str(eval_node(c, r, {0: True})) if r is not None else '0'})
        except (ValueError, KeyError) as ex:
            invalid.append({'case': name, 'accepted': False, 'error': str(ex)})
    result['compiler_invalid_inputs'] = invalid

finally:
    if sys.platform == 'win32':
        for lib in loaded_libs:
            try:
                ct.windll.kernel32.FreeLibrary(lib._handle)
            except Exception:
                pass
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

print(json.dumps(result, indent=2), flush=True)
failed = (
    any(x['mismatches'] for x in result['threshold_comparison']) or
    bad or
    any(x['exit'] for x in result['sanitizer']) or
    not all(result['regeneration'].values()) or
    any(x['accepted'] for x in invalid)
)
sys.exit(1 if failed else 0)
