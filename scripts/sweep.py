"""Run a declarative experiment spec (experiments/<id>.yaml) -- the one launcher.

    python scripts/sweep.py dry-run experiments/<id>.yaml    # print every cell's argv (DONE marked)
    python scripts/sweep.py status  experiments/<id>.yaml    # done / total per run
    python scripts/sweep.py run     experiments/<id>.yaml    # launch (run inside tmux); resumable
    echo pause|drain|resume > logs/<id>/control               # control a running sweep

See src/sweep/spec.py for the spec format and src/sweep/scheduler.py for the behaviour.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sweep.scheduler import Sweep  # noqa: E402


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ('run', 'status', 'dry-run'):
        print(__doc__)
        return 2
    sw = Sweep(sys.argv[2])
    {'run': sw.run, 'status': sw.status, 'dry-run': sw.dry_run}[sys.argv[1]]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
