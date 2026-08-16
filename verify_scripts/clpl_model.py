"""CLPL (Cour, Sapp & Taskar, JMLR 2011, "Learning from Partial Labels")
multiclass-to-binary reduction and the paper's p/q/epsilon controlled
ambiguity generator -- confirmed by direct PDF text extraction from
CLPL.pdf (see verify_scripts/clpl_verify.py for the quoted passages this
was derived from). This is CLPL-specific machinery, so it lives in this
CLPL-scoped file rather than a shared verify_scripts/models.py (other
paper-verification scripts are being added to verify_scripts/ in parallel
and should not collide on a generic filename).

Section 6 (p.1514, quoted, notation cleaned up):
    "We stack the parameters and features into one vector as follows
    below, so that g_a(x) = w_a . f(x) = w . f(x,a):
        w = [w_1; ...; w_L] ;  f(x,a) = [1(a=1)f(x); ...; 1(a=L)f(x)].
    We also define f(x,y) to be the average feature vector of the labels
    in the set y:  f(x,y) = (1/|y|) sum_{a in y} f(x,a).
    ...we simply transform the m partially labelled training examples
    S = {x_i,y_i} into m positive examples S+ = {f(x_i,y_i)} and
    sum_i (L - |y_i|) negative examples S- = {f(x_i,a)}_{i, a not in y_i}.
    ...We use the off-the-shelf implementation of binary SVM with squared
    hinge (Fan et al., 2008) in most of our experiments, where the
    objective is:
        min_w (1/2)||w||_2^2 + C sum_i max(0, 1 - w.f(x_i,y_i))^2
                              + C sum_{i,a not in y_i} max(0, 1 + w.f(x_i,a))^2."
(Fan et al., 2008 is LIBLINEAR; sklearn.svm.LinearSVC(loss='squared_hinge',
penalty='l2') is the standard Python equivalent of that solver.)

ASSUMPTIONS (flagged per task instructions rather than silently guessed):
  - No bias/intercept term. The paper's g_a(x) = w_a . f(x) and the SVM
    objective quoted above never introduce an offset/bias -- so the
    reduction is fit with LinearSVC(fit_intercept=False), not sklearn's
    default (fit_intercept=True). Using the default would silently add
    model capacity the paper's formulation doesn't have.
  - Regularization constant C=1.0 (sklearn's default). The paper does not
    report a specific value or a search procedure for C anywhere in the
    text (checked); C=1.0 is used as a reasonable, undocumented-elsewhere
    default rather than a paper-confirmed value.
  - Multiclass prediction is argmax_a w_a . f(x) over ALL L classes (the
    paper's *inductive* test-time rule, Section 7.3: "a* = h(x) = argmax
    a in {1..L} g_a(x)" -- as opposed to the transductive rule, which
    restricts the argmax to the ambiguous label set y). This repo's
    verification script only implements the inductive protocol.

Section 7.1 / Table 3 (p.1518-1519, quoted) defines the controlled
ambiguity generator used for the UCI experiments:
    "p represents the proportion of examples that are ambiguously
    labeled. q represents the number of extra labels for each ambiguous
    example (generated uniformly without replacement)."
Table 3 lists ecoli / derma / abalone under the "ambiguity size" sweep,
i.e. varying q at p=1, with q ranging over [0, 0.9(L-1)] -- generate_pqe
below implements exactly this "ambiguity size" (p, q) mode. The paper's
separate "degree of ambiguity" (epsilon) mode is used only for the FIW
face-recognition sweeps, not for the 3 UCI datasets this script targets,
but is implemented too (generate_pqe(..., epsilon=...)) for completeness,
per footnote 3 (p.1518, quoted): "We first choose at random for each
label a dominant co-occurring label which is sampled with probability
epsilon; the rest of the labels are sampled uniformly with probability
(1 - epsilon)/(L - 2) (there is a single extra label per example)."
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# p/q/epsilon controlled ambiguity generator (Section 7.1 / Table 3)
# ---------------------------------------------------------------------------

def generate_pqe(y_true: np.ndarray, L: int, p: float, q: float,
                  rng: np.random.Generator, epsilon: float | None = None):
    """Generate candidate label sets under the paper's controlled ambiguity
    model.

    Args:
        y_true: (m,) int array of true labels in [0, L).
        L: total number of classes.
        p: probability an example receives an ambiguous (>1 label) candidate
            set at all (Table 3's "p").
        q: number of extra false labels added when an example IS ambiguous,
            drawn uniformly without replacement from the L-1 wrong classes
            (Table 3's "q", "ambiguity size" mode). Ignored if epsilon is
            given (that mode always adds exactly 1 extra label).
        rng: numpy random Generator for reproducibility.
        epsilon: if given, switches to the paper's "degree of ambiguity"
            mode (footnote 3): a per-class dominant co-occurring label is
            sampled with probability epsilon, else a uniformly random other
            label is sampled with probability (1-epsilon)/(L-2); always
            exactly one extra label. Requires L >= 3.

    Returns:
        List of length m of 1-D int np.ndarray candidate label sets (each
        contains the true label plus 0 or more extra labels, no duplicates).
    """
    m = len(y_true)
    ambiguous_mask = rng.random(m) < p
    all_classes = np.arange(L)

    dominant = None
    if epsilon is not None:
        if L < 3:
            raise ValueError("epsilon-mode ambiguity requires L >= 3")
        # Fixed per-class "dominant co-occurring label" (one per true class),
        # sampled once for this generator call, reused across examples.
        dominant = np.empty(L, dtype=np.int64)
        for c in range(L):
            others = all_classes[all_classes != c]
            dominant[c] = rng.choice(others)

    q_eff = int(max(0, min(round(q), L - 1))) if epsilon is None else 1

    candidates = []
    for i in range(m):
        true_label = int(y_true[i])
        if not ambiguous_mask[i] or q_eff == 0:
            candidates.append(np.array([true_label], dtype=np.int64))
            continue
        if epsilon is not None:
            others = all_classes[all_classes != true_label]
            if rng.random() < epsilon:
                extra = np.array([dominant[true_label]], dtype=np.int64)
            else:
                non_dominant = others[others != dominant[true_label]]
                extra = np.array([rng.choice(non_dominant)], dtype=np.int64)
        else:
            others = all_classes[all_classes != true_label]
            extra = rng.choice(others, size=q_eff, replace=False)
        cand = np.unique(np.concatenate(([true_label], extra))).astype(np.int64)
        candidates.append(cand)
    return candidates


# ---------------------------------------------------------------------------
# Multiclass -> binary reduction (Section 6)
# ---------------------------------------------------------------------------

def _build_positive_matrix(X: np.ndarray, candidate_sets, L: int) -> np.ndarray:
    """f(x_i, y_i) = (1/|y_i|) * sum_{a in y_i} f(x_i, a) -- one row per
    training example, block-sparse (only the blocks for a in y_i are
    nonzero, each holding x_i / |y_i|)."""
    m, d = X.shape
    pos = np.zeros((m, d * L), dtype=np.float64)
    for i in range(m):
        yi = candidate_sets[i]
        val = X[i] / len(yi)
        for a in yi:
            a = int(a)
            pos[i, a * d:(a + 1) * d] = val
    return pos


def _build_negative_matrix(X: np.ndarray, candidate_sets, L: int) -> np.ndarray:
    """f(x_i, a) for every i and every a not in y_i -- one row per
    (example, non-candidate class) pair, block-sparse (only block a
    nonzero, holding x_i)."""
    m, d = X.shape
    row_x, row_class = [], []
    for i in range(m):
        yi_set = set(int(a) for a in candidate_sets[i])
        for a in range(L):
            if a not in yi_set:
                row_x.append(i)
                row_class.append(a)
    row_x = np.asarray(row_x, dtype=np.int64)
    row_class = np.asarray(row_class, dtype=np.int64)
    n_neg = len(row_x)
    neg = np.zeros((n_neg, d * L), dtype=np.float64)
    if n_neg == 0:
        return neg
    col_offsets = np.arange(d)
    cols = row_class[:, None] * d + col_offsets[None, :]
    rows = np.repeat(np.arange(n_neg)[:, None], d, axis=1)
    neg[rows, cols] = X[row_x]
    return neg


class CLPLClassifier:
    """Paper-exact CLPL: reduces the L-class partial-label problem to a
    single binary linear SVM with squared hinge loss (Section 6), then
    decomposes the learned weight vector back into per-class blocks for
    argmax prediction.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 20000,
                 random_state: int | None = None):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.W_ = None  # (L, d) per-class weight blocks
        self.d_ = None
        self.L_ = None

    def fit(self, X: np.ndarray, candidate_sets, L: int) -> "CLPLClassifier":
        m, d = X.shape
        pos = _build_positive_matrix(X, candidate_sets, L)
        neg = _build_negative_matrix(X, candidate_sets, L)
        X_bin = np.vstack([pos, neg])
        y_bin = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))])

        # loss='squared_hinge', penalty='l2' are sklearn's defaults and match
        # the paper's objective exactly; fit_intercept=False because the
        # paper's g_a(x) = w_a . f(x) has no bias term (see module docstring).
        clf = LinearSVC(C=self.C, loss='squared_hinge', penalty='l2',
                         fit_intercept=False, max_iter=self.max_iter,
                         random_state=self.random_state)
        clf.fit(X_bin, y_bin)
        w = clf.coef_.ravel()
        self.W_ = w.reshape(L, d)
        self.d_ = d
        self.L_ = L
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inductive-setting prediction: argmax_a w_a . f(x) over ALL L
        classes (Section 7.3)."""
        if self.W_ is None:
            raise RuntimeError("CLPLClassifier.predict called before fit")
        scores = X @ self.W_.T  # (n, L)
        return np.argmax(scores, axis=1)
