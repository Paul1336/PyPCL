"""scripts/sweep.py must reproduce an old hand-written launcher: same cells, same shard ids,
and argv that run_pipeline's own argparse parses to the same Namespace.

    python tests/sweep_equiv.py
Old launcher: logs/noctr_1118/launch.py (its grid code is executed with ROOT patched, up to the
point where it starts touching /proc). The temp wid.json is written under logs/_sweep_equiv.
"""
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


def old_jobs():
    src = open(os.path.join(ROOT, 'logs/noctr_1118/launch.py')).read()
    src = src.replace("ROOT = '/nfs/home/Paul/PyPCL/.claude/worktrees/switch-analysis'", f"ROOT = {ROOT!r}")
    src = src[:src.index('def gpu_procs')]  # grid / wid / order definitions only
    ns = {'__name__': 'old_noctr'}
    exec(compile(src, 'noctr_launch', 'exec'), ns)
    jobs = ns['order']
    return {(j['run'], j['a'], j['k'], j['s']): (j['wid'], ns['cmd_for'] if 'cmd_for' in ns else None, j) for j in jobs}, ns


def parse(argv):
    import run_pipeline
    import argparse
    parser = None
    # reuse run_pipeline's parser construction by intercepting parse_args
    captured = {}
    orig = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        captured['ns'] = orig(self, argv, namespace)
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        sys.argv = ['run_pipeline.py'] + argv
        try:
            run_pipeline.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    ns = vars(captured['ns'])
    ns.pop('data_dir', None)
    return ns


def main():
    from src.sweep import scheduler
    old, ns = old_jobs()
    tmp_logs = os.path.join(ROOT, 'logs', '_sweep_equiv')
    shutil.rmtree(tmp_logs, ignore_errors=True)
    sw = scheduler.Sweep.__new__(scheduler.Sweep)
    spec_path = os.path.join(ROOT, 'experiments/noctr_1118.yaml')
    import yaml
    spec = yaml.safe_load(open(spec_path))
    spec['id'] = '_sweep_equiv'
    p2 = os.path.join(ROOT, 'logs', '_sweep_equiv.yaml')
    os.makedirs(tmp_logs, exist_ok=True)
    yaml.safe_dump(spec, open(p2, 'w'), sort_keys=False)
    sw = scheduler.Sweep(p2)
    new = {(c.run_name, c.algorithm, c.k, c.seed): c for c in sw.cells}
    fails = []
    if set(new) != set(old):
        fails.append(f'cell sets differ: only old {len(set(old) - set(new))}, only new {len(set(new) - set(old))}')
    for key in set(new) & set(old):
        wid_old, _, j = old[key]
        c = new[key]
        if c.wid != wid_old:
            fails.append(f'{key}: wid {c.wid} vs {wid_old}')
            continue
        from src.sweep.spec import build_argv
        a_new = parse(build_argv(sw.spec, c, 'python')[2:])
        cmd = ns['cmd_for'](j) if 'cmd_for' in ns else None
        if cmd is None:
            src = open(os.path.join(ROOT, 'logs/noctr_1118/launch.py')).read()
            exec(compile(src[src.index('def cmd_for'):src.index('def logdir_for')], 'cmd_for', 'exec'), ns)
            cmd = ns['cmd_for'](j)
        a_old = parse(cmd[2:])
        if a_new != a_old:
            diff = {k: (a_new.get(k), a_old.get(k)) for k in set(a_new) | set(a_old) if a_new.get(k) != a_old.get(k)}
            fails.append(f'{key}: argv differs {diff}')
    shutil.rmtree(tmp_logs, ignore_errors=True)
    os.remove(p2)
    print(f'{len(new)} cells compared')
    for f in fails[:10]:
        print('FAIL', f)
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
