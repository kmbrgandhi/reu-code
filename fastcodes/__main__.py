"""Command-line interface. Run `python -m fastcodes --help`."""
import argparse
import json
import os
import sys

import numpy as np

from . import records, verify
from .problems import binary_from_hex, complementary_set, psl, type2
from .search import ALGOS, PRESETS, save, search, search_parallel, warmup


def _parse_params(items):
    out = {}
    for it in items or []:
        if '=' not in it:
            raise SystemExit(f'--param expects key=value, got {it!r}')
        k, v = it.split('=', 1)
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v  # named choices, e.g. schedule=lin_add
    return out


def _lengths(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _problem(kind, L, args):
    if kind == 'psl':
        return psl(L, args.phases, args.doppler, args.metric or 'psl', args.p)
    if kind == 'set':
        return complementary_set(L, args.codes, args.phases, args.metric or 'psl', args.p)
    return type2(L, args.codes, args.phases, args.metric or 'isl', args.p)


def _run_one(pb, args):
    params = _parse_params(args.param)
    if args.workers == 1:
        return search(pb, args.algo, args.time, args.target, args.seed, params)
    return search_parallel(pb, args.algo, args.time, args.workers, args.target, args.seed, params)


def _compare(L, peak, table):
    """One line comparing a binary PSL result with the best published value."""
    if L not in table:
        return '    published: no reference value in fastcodes/data for this length'
    ref = table[L]
    if round(peak) < ref['psl']:
        tag = '*** BELOW THE BEST PUBLISHED VALUE: check the code and the literature ***'
    elif round(peak) == ref['psl']:
        tag = 'ties the best published value'
    else:
        tag = f"{round(peak) - ref['psl']} above the best published value"
    line = f"    published: PSL {ref['psl']} ({ref['status']}; {ref['source']}) -> {tag}"
    if 'MISMATCH' in ref.get('note', '') or 'typo' in ref.get('note', ''):
        line += "\n    caution: the published code for this length does not reproduce its claimed PSL (see the note in fastcodes/data)"
    return line


def cmd_run(args):
    binary_psl = args.kind == 'psl' and args.phases == 2 and args.doppler == 1
    table = records.binary_best_known() if binary_psl else {}
    for L in _lengths(args.length):
        pb = _problem(args.kind, L, args)
        target = args.target
        if target is None and args.kind in ('set', 'type2'):
            target = 0.0
        if args.beat_record:
            if not binary_psl or L not in table:
                raise SystemExit(f'--beat-record needs a binary PSL length with a published value (L={L})')
            target = table[L]['psl'] - 1
        args_target = args.target
        args.target = target
        r = _run_one(pb, args)
        args.target = args_target
        print(r.summary())
        if binary_psl:
            print(_compare(L, r.peak, table))
        print('    code:', json.dumps(r.code))
        if args.out:
            save(r, args.out)


def cmd_report(args):
    """Best result per problem in a results file (and in checkpoints left by
    interrupted runs), compared with the records."""
    import glob
    best = {}
    entries = []
    if os.path.exists(args.file):
        with open(args.file) as f:
            entries += [(json.loads(line), False) for line in f if line.strip()]
    for ck in glob.glob(os.path.join(os.path.dirname(args.file) or '.', 'checkpoints', '*.json')):
        with open(ck) as f:
            entries.append((json.load(f), True))
    for d, from_ck in entries:
        d['_checkpoint'] = from_ck
        pr = d['problem']
        key = (pr['kind'], pr['L'], pr['K'], pr['phases'], pr['doppler'])
        if key not in best or d['peak'] < best[key]['peak']:
            best[key] = d
    table = records.binary_best_known()
    for key in sorted(best):
        d = best[key]
        kind, L, K, m, N = key
        line = (f"{kind:10s} L={L:4d} K={K:2d} m={m} N={N}  best peak {d['peak']:.6g} "
                f"[{'verified' if d['verified'] else 'NOT VERIFIED'}] via {d['algo']}"
                f"{' (from a checkpoint: run in progress or interrupted)' if d['_checkpoint'] else ''}")
        print(line)
        if kind == 'bin_psl':
            print(_compare(L, d['peak'], table))
        elif kind in ('bin_type2', 'cplx_type2') and d['peak'] == 0:
            print('    type-2 alternating set found:', json.dumps(d['code']))


def cmd_algos(args):
    for name, a in ALGOS.items():
        kind = 'continuous' if a.continuous else ('any' if a.both else 'discrete')
        print(f'{name:14s} [{kind}] {a.doc}')
        print('    ' + ', '.join(f'{n}={d:g}' for n, d in zip(a.names, a.defaults)))
    print('\npresets (2017 variant names):')
    for name, (base, over) in PRESETS.items():
        print(f'  {name:20s} = {base} {over}')


def cmd_check(args):
    L = args.length
    if args.kind == 'psl':
        c = binary_from_hex(args.hex[0], L)
        print(f'PSL over {args.doppler} Doppler bin(s): {verify.doppler_psl(c, args.doppler):g}')
        return
    codes = np.array([binary_from_hex(h, L) for h in args.hex])
    print(f'{len(codes)} codes of length {L}')
    print(f'  summed-autocorrelation peak (0 = complementary set): {verify.set_peak(codes):g}')
    print(f'  type-2 peak (0 = Sulzer type-2 alternating set):     {verify.type2_peak(codes):g}')
    print(f'  Lehtinen weak violation (0 = type-1 alternating):    {verify.lehtinen_weak_violation(codes):g}')


def cmd_records(args):
    t = records.binary_min_psl()
    print('binary minimum PSL (fastcodes/data/binary_min_psl.tsv):')
    for L in sorted(t):
        r = t[L]
        print(f"  L={L:3d}  PSL {r['psl']}  {r['status']}  [{r['source']}]{'  NOTE: ' + r['note'] if 'typo' in r['note'] else ''}")
    print('\npublished type-2 alternating sets (8 codes each):')
    for N, S in records.type2_known_sets().items():
        print(f'  length {N}: type-2 peak {verify.type2_peak(S):g}')


def cmd_barker(args):
    from . import barker as BK
    for L in _lengths(args.length):
        params = dict(margin=args.margin)
        if args.workers == 1:
            r = BK.search(L, args.time, args.seed, **params)
        else:
            r = BK.search_parallel(L, args.time, args.workers, args.seed, **params)
        cert = r.get('certification', {})
        status = ('CERTIFIED Barker sequence' if cert.get('certified')
                  else ('found but NOT certified' if r['found'] else 'not found'))
        print(f"barker N={L}: {status} | best interior max|C| = {r['interior_peak']:.9f} | "
              f"{r['time']:.0f}s, {r['runs']} runs, {r['level_descents']} level descents, "
              f"{r.get('workers', 1)} worker(s)")
        if cert:
            print(f"    interval-arithmetic upper bound on max|C|: {cert['max_c_upper']:.12f}")
        if args.out:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'a') as f:
                f.write(json.dumps(r, default=float) + '\n')


def cmd_barker_certify(args):
    from . import barker as BK
    if args.known:
        for N, ph in BK.known_codes().items():
            c = BK.certify(radians=ph)
            print(f"Nunn & Coxson 2009, N={N}: max|C| <= {c['max_c_upper']:.9f} certified={c['certified']}")
        return
    with open(args.file) as f:
        for line in f:
            r = json.loads(line)
            if 'm' not in r:
                continue
            c = BK.certify(m=r['m'], H=r.get('H', BK.H_DEFAULT))
            print(f"N={r['N']}: max|C| <= {c['max_c_upper']:.12f} certified={c['certified']}")


def cmd_exhaust(args):
    from . import exhaust
    for L in _lengths(args.length):
        t = __import__('time').time()
        r = exhaust.search(L, batches_hint=200, progress=True)
        print(f"exhaustive type-2, N={L}, 8 codes: {r['solutions']} sets "
              f"({r['chirp_classes']} up to the chirp symmetry), all re-checked independently | "
              f"{r['search_s']:.0f}s search, {r['pairs']:,} vector pairs, {r['threads']} threads")
        if r['solutions'] == 0:
            print('    NO type-2 set of 8 binary codes exists at this length (complete search).')
        for S in r['solution_codes'][: args.show]:
            print('    hex:', [format(int(''.join('0' if b > 0 else '1' for b in c), 2), 'X').zfill((L + 3) // 4) for c in S])
        if args.out:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'a') as f:
                f.write(json.dumps({k: v for k, v in r.items() if k != 'solution_indices'}, default=float) + '\n')


def cmd_vsearch(args):
    from . import vsearch
    for L in _lengths(args.length):
        r = vsearch.search_parallel(L, args.codes, args.time, args.workers,
                                    composition=not args.no_composition)
        tag = 'FOUND type-2 set' if r['found'] else 'not found'
        print(f"vector-space search, N={L}, {args.codes} codes: {tag} | closest peak {r['peak']:g} "
              f"(verified {r['verified_peak']:g}) | {r['time']:.0f}s, {r['runs']} runs, {r['workers']} workers")
        if r['found']:
            print('    hex:', r['hex'])
        if args.out:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'a') as f:
                f.write(json.dumps(r, default=float) + '\n')


def cmd_bench(args):
    from . import bench
    bench.main(quick=args.quick)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m fastcodes', description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help='search for codes')
    r.add_argument('kind', choices=['psl', 'set', 'type2'],
                   help='psl: one code; set: complementary set; type2: type-2 alternating set')
    r.add_argument('--length', required=True, help='e.g. 64, or 106-120, or 13,15,17')
    r.add_argument('--codes', type=int, default=8, help='number of codes (set, type2)')
    r.add_argument('--phases', type=int, default=2, help='2 binary, m>=3 m-phase, 0 continuous')
    r.add_argument('--doppler', type=int, default=1, help='Doppler bins over the full band (psl)')
    r.add_argument('--metric', choices=['psl', 'isl', 'pnorm'], help='default psl (type2: isl)')
    r.add_argument('--p', type=float, default=4.0, help='exponent for --metric pnorm')
    r.add_argument('--algo', default='sdls', choices=sorted(ALGOS) + sorted(PRESETS))
    r.add_argument('--time', type=float, default=60.0, help='seconds per length')
    r.add_argument('--workers', type=int, default=0, help='processes (default: all cores)')
    r.add_argument('--target', type=float, help='stop when the peak is <= target')
    r.add_argument('--beat-record', action='store_true',
                   help='binary PSL: stop early only if the published best value is beaten')
    r.add_argument('--seed', type=int)
    r.add_argument('--param', action='append', help='algorithm parameter key=value (repeatable)')
    r.add_argument('--out', default='results/runs.jsonl', help='append results here ("" to skip)')
    r.set_defaults(func=cmd_run)

    sub.add_parser('algos', help='list algorithms, parameters, and presets').set_defaults(func=cmd_algos)

    rp = sub.add_parser('report', help='best result per problem in a results file')
    rp.add_argument('file', nargs='?', default='results/runs.jsonl')
    rp.set_defaults(func=cmd_report)

    c = sub.add_parser('check', help='verify codes given in hex (+1 -> 0 bit, first element = MSB)')
    c.add_argument('kind', choices=['psl', 'set'], help='psl: one code; set: a code set (all checks)')
    c.add_argument('--length', type=int, required=True)
    c.add_argument('--doppler', type=int, default=1)
    c.add_argument('hex', nargs='+')
    c.set_defaults(func=cmd_check)

    sub.add_parser('records', help='print the published reference values').set_defaults(func=cmd_records)

    ex = sub.add_parser('exhaust-type2', help='complete search for type-2 sets of 8 binary codes')
    ex.add_argument('--length', required=True, help='e.g. 13 (each extra length costs about 16x more)')
    ex.add_argument('--show', type=int, default=3, help='how many sets to print')
    ex.add_argument('--out', default='results/exhaust.jsonl')
    ex.set_defaults(func=cmd_exhaust)

    vs = sub.add_parser('vsearch', help='fast vector-space search for type-2 sets')
    vs.add_argument('--length', required=True)
    vs.add_argument('--codes', type=int, default=8)
    vs.add_argument('--time', type=float, default=600.0, help='seconds per length')
    vs.add_argument('--workers', type=int, default=0, help='processes (default: all cores)')
    vs.add_argument('--no-composition', action='store_true',
                    help="don't fix the forced type composition (K/4 A + K/2 B + K/4 C)")
    vs.add_argument('--out', default='results/vsearch.jsonl')
    vs.set_defaults(func=cmd_vsearch)

    bk = sub.add_parser('barker', help='search for unimodular (polyphase) Barker sequences')
    bk.add_argument('--length', required=True, help='e.g. 71, or 71,73-75')
    bk.add_argument('--time', type=float, default=600.0, help='seconds per length')
    bk.add_argument('--workers', type=int, default=0, help='processes (default: all cores)')
    bk.add_argument('--margin', type=float, default=1e-7, help='require max|C| <= 1 - margin')
    bk.add_argument('--seed', type=int)
    bk.add_argument('--out', default='results/barker.jsonl')
    bk.set_defaults(func=cmd_barker)

    bc = sub.add_parser('barker-certify', help='interval-arithmetic check of Barker results')
    bc.add_argument('file', nargs='?', default='results/barker.jsonl')
    bc.add_argument('--known', action='store_true', help='check the published Nunn & Coxson codes')
    bc.set_defaults(func=cmd_barker_certify)

    b = sub.add_parser('bench', help='speed benchmarks against the 2017 evaluation approach')
    b.add_argument('--quick', action='store_true')
    b.set_defaults(func=cmd_bench)

    sub.add_parser('warmup', help='compile everything once (takes a minute or two)').set_defaults(
        func=lambda a: (warmup(), print('compiled and cached')))

    args = ap.parse_args(argv)
    if getattr(args, 'workers', None) == 0:
        import os
        args.workers = os.cpu_count()
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
