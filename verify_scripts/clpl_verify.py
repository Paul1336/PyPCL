"""Paper-exact verification of CLPL (Cour, Sapp & Taskar, "Learning from
Partial Labels", JMLR 2011) on the paper's own 3 UCI benchmarks
(dermatology, ecoli, abalone), reproducing:

  - Section 6's multiclass-to-binary-SVM reduction (implemented in
    verify_scripts/clpl_model.py: CLPLClassifier), trained with a linear
    SVM + squared hinge loss (LIBLINEAR / sklearn.svm.LinearSVC), the
    paper's own "off-the-shelf" solver choice (Fan et al., 2008).
  - Section 7.1/Table 3's p/q controlled ambiguity generator
    (verify_scripts/clpl_model.py: generate_pqe), in the "ambiguity size"
    mode the paper itself uses for these exact 3 datasets (Table 3, rows
    "ecoli"/"derma"/"abalone": p=1, q in [0, 0.9(L-1)]). This script fixes
    q = round((L-1)/2), a representative "medium ambiguity" point within
    that paper-used sweep range (the user did not request the full sweep,
    see verify_scripts/clpl_model.py docstring for the exact quoted table).
  - Section 7.2.1's UCI dataset choice + feature standardization ("each
    feature was independently scaled to have zero mean and unit variance"):
    reuses src/pipeline/datasets/uci_tabular.py's `_load_ucirepo` loader
    (same ucimlrepo ids: dermatology=33, ecoli=39, abalone=1; same
    mean-impute + z-score standardization), NOT its `build_tabular_loaders`
    candidate generator, which is a fixed-k scheme unrelated to this
    paper's p/q model.
  - Section 7.3's inductive evaluation protocol: 50/50 random train/test
    split, 20 trials (each trial redraws both the split and the ambiguous
    labels), mean +/- std 0/1 test error (reported here as accuracy).

Usage:
    python verify_scripts/clpl_verify.py [--seed 42] [--n_trials 20]

Output: appends one row per dataset to verify_results/clpl.csv (created
with a header if it doesn't exist yet).

No GPU dependency -- CLPL reduces to a linear SVM (CPU-only, sklearn).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clpl_model import CLPLClassifier, generate_pqe  # noqa: E402
from src.pipeline.datasets.uci_tabular import _load_ucirepo, _UCI_IDS  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402

DATASETS = ["dermatology", "ecoli", "abalone"]

CSV_HEADER = [
    "dataset", "config", "seed", "n_trials", "mean_accuracy", "std_accuracy",
    "paper_target_accuracy", "training_time_s", "timestamp", "notes",
]

# The paper's Table 3 lists ecoli/derma/abalone under the "ambiguity size"
# sweep (p=1, q in [0, 0.9(L-1)]), reported as a curve in Figure 8 -- not a
# single tabulated accuracy number at any specific q. Since q=(L-1)/2 (this
# script's chosen point) is not a value the paper prints as text/a table
# cell, there is no number to compare against without reading pixel
# positions off a plot -- left blank per the task instructions rather than
# fabricated.
PAPER_TARGET_ACCURACY = {"dermatology": "", "ecoli": "", "abalone": ""}


def run_dataset(name: str, base_seed: int, n_trials: int) -> dict:
    data = _load_ucirepo(name, _UCI_IDS[name])
    X = data["X"].astype(np.float64)
    y = data["y"]
    L = data["n_classes"]
    q = int(round((L - 1) / 2))
    p = 1.0

    accs = []
    t0 = time.time()
    for t in range(n_trials):
        trial_seed = base_seed + t
        rng = np.random.default_rng(trial_seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=trial_seed, shuffle=True,
        )
        candidates = generate_pqe(y_train, L, p=p, q=q, rng=rng)
        clf = CLPLClassifier(C=1.0, random_state=trial_seed)
        clf.fit(X_train, candidates, L)
        preds = clf.predict(X_test)
        acc = float(np.mean(preds == y_test))
        accs.append(acc)
        print(f"  [{name}] trial {t + 1}/{n_trials}: accuracy={acc * 100:.2f}%", flush=True)
    elapsed = time.time() - t0

    accs = np.array(accs)
    mean_acc, std_acc = float(accs.mean()), float(accs.std())
    config = f"p=1,q={q},C=1.0"
    print(f"[CLPL] dataset={name} mean_accuracy={mean_acc * 100:.2f}%±{std_acc * 100:.2f}%", flush=True)

    return {
        "dataset": name,
        "config": config,
        "seed": base_seed,
        "n_trials": n_trials,
        "mean_accuracy": f"{mean_acc:.6f}",
        "std_accuracy": f"{std_acc:.6f}",
        "paper_target_accuracy": PAPER_TARGET_ACCURACY.get(name, ""),
        "training_time_s": f"{elapsed:.2f}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": (
            f"L={L} (live ucimlrepo class count), q=round((L-1)/2)={q}; "
            "inductive protocol, argmax over all L classes; fit_intercept=False "
            "(paper's g_a(x)=w_a.f(x) has no bias term); C=1.0 is sklearn's default, "
            "paper states no specific C or search procedure; paper's Table 3/Fig 8 "
            "reports this dataset's ambiguity-size sweep as a q-in-[0,0.9(L-1)] curve, "
            "not a tabulated accuracy at q=(L-1)/2 -> paper_target_accuracy not directly comparable."
        ),
    }


def append_results(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    is_new = not os.path.isfile(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-exact CLPL (Cour, Sapp & Taskar, JMLR 2011) verification "
                     "on dermatology/ecoli/abalone UCI benchmarks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (default: 42).")
    parser.add_argument("--n_trials", type=int, default=20,
                         help="Number of random split/ambiguity trials per dataset "
                              "(paper uses 20; default: 20).")
    args = parser.parse_args()

    out_path = os.path.join(_REPO_ROOT, "verify_results", "clpl.csv")
    rows = []
    for name in DATASETS:
        print(f"=== CLPL verification: {name} ===", flush=True)
        rows.append(run_dataset(name, args.seed, args.n_trials))
    append_results(rows, out_path)
    print(f"\nResults appended to {out_path}")


if __name__ == "__main__":
    main()
