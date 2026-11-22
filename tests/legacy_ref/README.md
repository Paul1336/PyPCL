Frozen copy of `src/` at commit 506c899 (pre-refactor, after the Phase 0 bug fixes), with imports
rewritten `src.` -> `legacy_src.`. Used ONLY by tests/step_equiv.py as the reference implementation
for step-level equivalence (loss values, gradients, one-step state updates). Never import it from
production code. Delete once the refactor is finished and the goldens are trusted.
