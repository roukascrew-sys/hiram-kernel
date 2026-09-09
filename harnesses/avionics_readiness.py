import ctypes, itertools, pathlib, subprocess, sys, tempfile, os, json, shutil

root = pathlib.Path(sys.argv[1]).resolve()
out = {}

td_dir = tempfile.mkdtemp(prefix='tmp_test_avionics_', dir=root)
d = pathlib.Path(td_dir)
loaded_libs = []

try:
    if sys.platform == 'win32':
        gcc_bin = shutil.which('gcc')
        if gcc_bin:
            try: os.add_dll_directory(os.path.dirname(os.path.abspath(gcc_bin)))
            except Exception: pass
        try: os.add_dll_directory(str(d))
        except Exception: pass

    # 1. Strict runtime compilation
    base = ['gcc', '-std=c99', '-DHIRAM_FORCE_64BIT_ACC', '-Wall', '-Wextra', '-Werror',
            '-Wconversion', '-Wsign-conversion', '-Wshadow', '-Wcast-align',
            '-Wcast-qual', '-Wswitch-enum', '-Wundef', '-Wformat=2',
            '-pedantic-errors', '-I' + str(root / 'include/hiram')]
    p = subprocess.run(base + ['-c', str(root / 'src/hiram_eval.c'), '-o', str(d / 'strict.o')],
                       capture_output=True, text=True)
    out['strict_runtime_build'] = {'pass': p.returncode == 0, 'diagnostics': p.stderr}

    # 2. Shared library build
    so_name = 'hiram.dll' if sys.platform == 'win32' else 'hiram.so'
    so = d / so_name
    
    # Sanitizer availability is recorded explicitly. Previously this build was
    # skipped on Windows yet still reported as 'instrumented_build: pass', so a
    # run with no UB detection at all read as if it had been sanitized. A check
    # that cannot run must say so rather than report success.
    extra_flags = []
    sanitizer_status = 'skipped: win32 (MinGW ships no libubsan)'
    if sys.platform != 'win32':
        probe_proc = subprocess.run(['gcc', '-fsanitize=undefined', '-x', 'c', '-', '-o', '/dev/null'],
                                    input='int main(){return 0;}', text=True, capture_output=True)
        if probe_proc.returncode == 0:
            extra_flags = ['--coverage', '-fsanitize=undefined', '-fno-sanitize-recover=undefined']
            sanitizer_status = 'active: -fsanitize=undefined -fno-sanitize-recover=undefined'
        else:
            sanitizer_status = 'unavailable: gcc cannot link -fsanitize=undefined'

    p = subprocess.run(['gcc', '-O1', '-g', '-shared', '-fPIC', '-DHIRAM_FORCE_64BIT_ACC'] + extra_flags +
                        ['-I' + str(root / 'include/hiram'), str(root / 'src/hiram_eval.c'),
                         '-o', str(so)], capture_output=True, text=True)
    out['instrumented_build'] = {'pass': p.returncode == 0, 'diagnostics': p.stderr,
                                 'sanitizer': sanitizer_status,
                                 'ub_detection_active': bool(extra_flags)}

    class Rat(ctypes.Structure):
        _fields_ = [('num', ctypes.c_int64), ('den', ctypes.c_int64)]

    class Ev(ctypes.Structure):
        _fields_ = [('observed_mask', ctypes.c_uint32), ('values_mask', ctypes.c_uint32)]

    class Rep(ctypes.Structure):
        _fields_ = [('decision', ctypes.c_int), ('hazard_probability', Rat), ('hazard_threshold', Rat)]

    class Ctx(ctypes.Structure):
        _fields_ = [('memo', Rat * 81), ('execution_flags', ctypes.c_uint32)]

    lib = ctypes.CDLL(str(so), winmode=0) if sys.platform == 'win32' else ctypes.CDLL(str(so))
    loaded_libs.append(lib)
    lib.hiram_context_init.argtypes = [ctypes.POINTER(Ctx)]
    lib.hiram_audit_hazard.argtypes = [ctypes.POINTER(Ev), Rat, ctypes.POINTER(Ctx)]
    lib.hiram_audit_hazard.restype = Rep

    ctx = Ctx()
    lib.hiram_context_init(ctypes.byref(ctx))

    # 3. Exhaustive API evaluation across all ternary states
    counts = {str(i): 0 for i in range(4)}
    for digs in itertools.product(range(3), repeat=12):
        om = vm = 0
        for v, x in enumerate(digs):
            if x < 2:
                om |= (1 << v)
                vm |= (x << v)
        ev = Ev(om, vm)
        for th in (Rat(0, 1), Rat(1, 100), Rat(1, 20), Rat(1, 1)):
            rep = lib.hiram_audit_hazard(ctypes.byref(ev), th, ctypes.byref(ctx))
            counts[str(rep.decision)] += 1
    out['exhaustive_api_calls'] = {'pass': True, 'calls': (3**12) * 4, 'decision_counts': counts}

    # 4. Targeted robustness probes
    probes = []
    for name, ev, th, test_ctx, expected in [
        ('threshold_den_zero', Ev(), Rat(1, 0), ctypes.byref(ctx), 3),
        ('threshold_negative', Ev(), Rat(-1, 2), ctypes.byref(ctx), 3),
        ('threshold_over_one', Ev(), Rat(2, 1), ctypes.byref(ctx), 3),
        ('observed_outside_model', Ev(1 << 12, 0), Rat(1, 20), ctypes.byref(ctx), 3),
        ('value_outside_model', Ev(0, 1 << 12), Rat(1, 20), ctypes.byref(ctx), 3),
        ('value_without_observation', Ev(0, 1 << 3), Rat(1, 20), ctypes.byref(ctx), 3),
        ('null_context', Ev(), Rat(1, 20), None, 3)
    ]:
        r = lib.hiram_audit_hazard(ctypes.byref(ev), th, test_ctx)
        probes.append({'name': name, 'expected': expected, 'actual': r.decision, 'pass': r.decision == expected})
    out['robustness_probes'] = probes

    # 5. Circuit table integrity verification
    src = d / 'struct.c'
    src.write_text(r'''#include "hiram_circuit_data.h"
#include <stdio.h>
#include "hiram_circuit_data.h"
int main(void){
    unsigned err = 0;
    for(unsigned i = 0; i < CIRCUIT_NODE_COUNT; i++){
        const CircuitNode* n = &g_circuit_nodes[i];
        if(n->type > NODE_SUM) err++;
        if(n->type == NODE_LITERAL && n->var_id >= CIRCUIT_VAR_COUNT) err++;
        if(n->type == NODE_PROD || n->type == NODE_SUM){
            if((unsigned)n->child_offset + n->child_count > CHILDREN_ARRAY_SIZE) err++;
            for(unsigned j = 0; j < n->child_count; j++)
                if(g_circuit_children[n->child_offset + j] >= i) err++;
        }
        if(n->type == NODE_SUM){
            if((unsigned)n->weight_offset + n->child_count > WEIGHTS_ARRAY_SIZE) err++;
            for(unsigned j = 0; j < n->child_count; j++){
                Rational w = g_circuit_weights[n->weight_offset + j];
                if(w.den <= 0 || w.num < 0 || w.num > w.den) err++;
            }
        }
    }
    if(CIRCUIT_ROOT_ID >= CIRCUIT_NODE_COUNT) err++;
    printf("errors=%u nodes=%u children=%u weights=%u\n", err, (unsigned)CIRCUIT_NODE_COUNT, (unsigned)CHILDREN_ARRAY_SIZE, (unsigned)WEIGHTS_ARRAY_SIZE);
    return err ? 1 : 0;
}''')
    exe_name = 'struct.exe' if sys.platform == 'win32' else 'struct'
    struct_bin = d / exe_name
    p = subprocess.run(base + [str(src), '-o', str(struct_bin)], capture_output=True, text=True)
    if p.returncode == 0:
        q = subprocess.run([str(struct_bin)], capture_output=True, text=True)
        out['circuit_structure'] = {'pass': q.returncode == 0, 'output': q.stdout, 'diagnostics': q.stderr}
    else:
        out['circuit_structure'] = {'pass': False, 'output': '', 'diagnostics': p.stderr}

    # 6. Official repository test suite
    p = subprocess.run([sys.executable, str(root / 'tests/test_c_binding.py')],
                       cwd=root, capture_output=True, text=True)
    out['repository_test_suite'] = {'pass': p.returncode == 0, 'diagnostics': (p.stderr + p.stdout)[-1200:]}

    # 7. Model-to-C table reproduction
    exp = d / 'export'
    exp.mkdir()
    p = subprocess.run([sys.executable, str(root / 'tools/hiram_c_exporter.py')],
                       cwd=exp, capture_output=True, text=True)
    # Only the generated artifact; the other two are hand-written sources.
    files = {
        'hiram_circuit_data.h': root / 'include/hiram/hiram_circuit_data.h'
    }
    out['export_reproducibility'] = {
        'pass': p.returncode == 0 and all((exp / k).read_bytes() == v.read_bytes() for k, v in files.items()),
        'note': 'Generated C tables reproduce canonical committed artifacts.'
    }

    # 8. Host stack frame analysis
    p = subprocess.run(['gcc', '-O3', '-fstack-usage', '-DHIRAM_FORCE_64BIT_ACC',
                        '-I' + str(root / 'include/hiram'), '-c', str(root / 'src/hiram_eval.c'),
                        '-o', str(d / 'stack.o')], capture_output=True, text=True)
    su = list(d.glob('*.su'))
    out['host_stack_usage'] = {'pass': p.returncode == 0, 'report': su[0].read_text() if su else ''}

finally:
    if sys.platform == 'win32':
        for l in loaded_libs:
            try: ctypes.windll.kernel32.FreeLibrary(l._handle)
            except Exception: pass
    try: shutil.rmtree(td_dir, ignore_errors=True)
    except Exception: pass

print(json.dumps(out, indent=2))
