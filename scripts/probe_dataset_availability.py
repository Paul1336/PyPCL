"""Phase 0 of the paper-original-benchmark dataset support effort
(see docs/dataset_availability_report.md and the plan in
docs/00_paper_alignment_guide.md).

Does cheap HTTP HEAD/GET probes (status code, content-length, content-type
only -- no parsing, no heavy dependencies) against every dataset URL needed
for the 6 already-verified papers' original benchmarks, so later phases know
which real-world data sources are actually alive before any loader code is
written around them.

Usage:
    python scripts/probe_dataset_availability.py [--out docs/dataset_availability_report.md]
"""

import argparse
import datetime

import requests

USER_AGENT = 'Mozilla/5.0 (compatible; PyPCL-dataset-probe/1.0)'
TIMEOUT = 20

# (name, url, note)
TARGETS = [
    # Tier 1: torchvision built-in image datasets (sanity-checked by import, not HTTP probe)
    # -- handled separately in check_torchvision_builtins().

    # Tier 2: CUB-200-2011 mirrors. The old vision.caltech.edu path is dead (404); the
    # dataset moved to data.caltech.edu's CaltechDATA repository (Zenodo-like), which
    # redirects to a presigned, short-TTL S3 URL -- HEAD on that URL alone isn't a
    # reliable probe (signature is timestamped), so the record page + HF mirror are
    # the actual availability signal.
    ('CUB-200-2011 (Caltech official, old path -- known dead)',
     'https://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz',
     'Phase 3 -- superseded, kept here to document the dead old URL'),
    ('CUB-200-2011 (CaltechDATA record page, current official source)',
     'https://data.caltech.edu/records/65de6-vp158',
     'Phase 3 primary source -- redirects to a presigned download URL, ~1.1GB'),
    ('CUB-200-2011 (HuggingFace mirror)',
     'https://huggingface.co/datasets/bentrevett/caltech-ucsd-birds-200-2011',
     'Phase 3 fallback mirror'),

    # Tier 3: classic real-world PLL datasets (Lost, MSRCv2, BirdSong, Soccer Player, Yahoo!News)
    # Exact filenames confirmed via mikigom/DNPL-PyTorch's README (which cites the same PALM-lab
    # source) -- Soccer Player and Yahoo! News have spaces/! in the filename, not CamelCase.
    ('PLL: Lost (.rar)', 'http://palm.seu.edu.cn/zhangml/files/lost.rar',
     'Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 936KB'),
    ('PLL: MSRCv2 (.rar)', 'http://palm.seu.edu.cn/zhangml/files/MSRCv2.rar',
     'Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 381KB'),
    ('PLL: BirdSong (.rar)', 'http://palm.seu.edu.cn/zhangml/files/BirdSong.rar',
     'Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 1.0MB'),
    ('PLL: Soccer Player (.rar)', 'http://palm.seu.edu.cn/zhangml/files/Soccer%20Player.rar',
     'Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 36MB'),
    ('PLL: Yahoo!News (.rar)', 'http://palm.seu.edu.cn/zhangml/files/Yahoo!%20News.rar',
     'Phase 4 -- PRODEN/MCL-LOG real-world benchmark; 28.7MB'),
    ('DNPL-PyTorch GitHub (source of exact filenames)', 'https://github.com/mikigom/DNPL-PyTorch',
     'Confirms the 5 PLL dataset filenames above'),

    # Tier 4: CLPL (Cour, Sapp & Taskar 2011) original data
    # tv_data.html confirmed to link tv_data.tar.gz (117MB, full: 8 LOST episodes,
    # ~3000 faces, lost_with_screenplay_supervision.mat with real screenplay-derived
    # ambiguous labels, and fiw_data.mat = LFW-derived "Faces in the Wild" subset) and
    # tv_data_small.tar.gz (13MB, 1 episode, for getting started).
    ('CLPL: tv_data.html (author page)', 'http://www.timotheecour.com/tv_data/tv_data.html',
     'Phase 4 -- CLPL original benchmark landing page'),
    ('CLPL: tv_data.tar.gz (full, 117MB)', 'http://www.timotheecour.com/tv_data/tv_data.tar.gz',
     'Phase 4 -- full CLPL dataset: raw 90x90x3 uint8 cropped face images + real candidate labels'),
    ('CLPL: tv_data_small.tar.gz (13MB)', 'http://www.timotheecour.com/tv_data/tv_data_small.tar.gz',
     'Phase 4 -- 1-episode starter version, used for local verification'),
]


def probe(name: str, url: str, note: str) -> dict:
    headers = {'User-Agent': USER_AGENT}
    try:
        r = requests.head(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code >= 400 or 'content-length' not in {k.lower() for k in r.headers}:
            # Some servers don't support HEAD properly; retry with a ranged GET.
            r = requests.get(url, headers={**headers, 'Range': 'bytes=0-0'}, timeout=TIMEOUT,
                              allow_redirects=True, stream=True)
        status = r.status_code
        length = r.headers.get('Content-Length', 'unknown')
        ctype = r.headers.get('Content-Type', 'unknown')
        final_url = r.url
        verdict = 'AVAILABLE' if status < 400 else f'DEAD (HTTP {status})'
    except requests.exceptions.RequestException as e:
        status, length, ctype, final_url = 'ERROR', 'n/a', 'n/a', url
        verdict = f'DEAD ({type(e).__name__}: {e})'
    return {
        'name': name, 'url': url, 'note': note, 'status': status,
        'length': length, 'content_type': ctype, 'final_url': final_url,
        'verdict': verdict,
    }


def check_torchvision_builtins() -> list:
    results = []
    try:
        import torchvision.datasets as tvd
        for cls_name in ['MNIST', 'FashionMNIST', 'KMNIST', 'SUN397']:
            available = hasattr(tvd, cls_name)
            results.append({
                'name': f'torchvision.datasets.{cls_name}', 'url': '(built into torchvision)',
                'note': 'Phase 1 -- built-in loader class', 'status': 'n/a', 'length': 'n/a',
                'content_type': 'n/a', 'final_url': '(built into torchvision)',
                'verdict': 'AVAILABLE (class exists)' if available else 'DEAD (class missing)',
            })
    except ImportError as e:
        results.append({
            'name': 'torchvision', 'url': '(local package)', 'note': 'Phase 1 prerequisite',
            'status': 'n/a', 'length': 'n/a', 'content_type': 'n/a', 'final_url': 'n/a',
            'verdict': f'DEAD (ImportError: {e})',
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='docs/dataset_availability_report.md')
    args = parser.parse_args()

    print('Checking torchvision built-in datasets...')
    results = check_torchvision_builtins()
    for r in results:
        print(f"  [{r['verdict']}] {r['name']}")

    print('\nProbing external dataset URLs...')
    for name, url, note in TARGETS:
        r = probe(name, url, note)
        results.append(r)
        print(f"  [{r['verdict']}] {name}  ({url})")

    lines = [
        '# Dataset Availability Report',
        '',
        f'Generated {datetime.datetime.now().isoformat(timespec="seconds")} by '
        '`scripts/probe_dataset_availability.py`. Cheap HTTP HEAD/GET probes only '
        '(status code / content-length / content-type) -- no file parsing, no schema '
        'validation. A verdict of AVAILABLE means the URL responds, not that the file '
        'has been downloaded and parsed successfully; DEAD means the URL did not resolve '
        'or returned an HTTP error at probe time.',
        '',
        '| Verdict | Name | URL | Status | Content-Length | Content-Type | Note |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in results:
        lines.append(
            f"| {r['verdict']} | {r['name']} | {r['url']} | {r['status']} | "
            f"{r['length']} | {r['content_type']} | {r['note']} |"
        )

    dead = [r for r in results if r['verdict'].startswith('DEAD')]
    lines += [
        '',
        f"## Summary: {len(results) - len(dead)}/{len(results)} available, {len(dead)} dead",
        '',
    ]
    if dead:
        lines.append('**Dead / unreachable at probe time:**')
        for r in dead:
            lines.append(f"- {r['name']}: {r['verdict']}")

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nReport written to {args.out}')


if __name__ == '__main__':
    main()
