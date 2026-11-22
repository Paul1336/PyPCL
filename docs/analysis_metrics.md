# Analysis metrics inventory (non-accuracy analyses)

Complete list of every analysis beyond test accuracy that appears in this project's experiments, organised as **where it originally lived → which function it now maps to in `src/analysis/`**.
Equivalence check: `python verify_scripts/check_analysis.py` (41 checks, all match the original implementations).

Conventions: Y = candidate set, Ȳ = complementary set, y = true label, B = batch, pool = `[q (B); k (B); queue (Q)]`.
"Ratio-of-sums" = numerator and denominator are each summed over the whole epoch, then divided; "mean-of-batch-means" = each batch is averaged first and every batch carries equal weight.

---

## A. Contrastive pair selection (`analysis.pairs`)

| Metric | Definition | Original source | New function |
|---|---|---|---|
| `pos_per_anchor` / `pos_nonkey_per_anchor` | Average number of positives selected per anchor (including / excluding its own key) | pcdiag `pair_stats` | `pool_pair_sums` + `pool_pair_metrics` |
| `pos_prec` / `pos_nonkey_prec` | Positive precision: share of selected positives with the same true label | same | same |
| `pos_recall` | Share of all same-class entries in the pool that get selected | same | same |
| `pos_viol_rate` / `pos_nonkey_viol_rate` | Share of positives whose two candidate sets are disjoint (impossible to be same class) | same | same |
| `fp_viol_share` | Share of false positives that a shared-candidate filter could remove | same | same |
| `pos_prec_if_filtered` | Precision after removing disjoint-candidate pairs (CandFilter upper bound) | same | same |
| `neg_per_anchor` / `neg_fn_rate` | Number of negatives / false-negative rate (negative impurity) | same | same |
| `neg_certified_rate` | Share of negatives with disjoint candidate sets (guaranteed different class) | same | same |
| `pos_precision` / `neg_precision` / tp fp tn fn / recall / cell % | Within-batch q×q block confusion (self excluded) | detail.py PiCO `--detail`, `final_epoch_pico_confusion` | `within_batch_confusion`, `confusion_ratios` |
| `precision_before/after`, `n_flipped` | PiCO-Oracle: randomly remove FPs until reaching the precision threshold | detail.py / oracle_pico_engine | `oracle_remove_correction` |
| `n_added`, `n_capped`, `add_ratio` | PiCO-Oracle-Add: randomly add TPs (with a cap) | same | `oracle_add_correction` |
| `neg_precision` (oracle) | Share of unselected pairs that are truly different class | same | `negative_precision` |
| ComCo `pos_precision` | Positive precision over the q+k block (2B, key included) | detail.py ComCo `--detail` | `pos_precision_2b` |
| PiCO+ `pos_prec` | Precision of the final SupCon mask (self excluded, kNN positives included) | picoplus.py | `supcon_mask_precision_sums` |

⚠ The three sets of pair statistics use different pair sets and **cannot be compared directly** (see the docstring at the top of `pairs.py`).

## B. Pseudo-labels / confidence / prototypes / selection (`analysis.labels`)

| Metric | Definition | Original source | New function |
|---|---|---|---|
| `pl_acc`, `pl_in_comp` | Pseudo-label accuracy; share of pseudo-labels that fall in Ȳ (PiCO: masked argmax; ComCo: unmasked) | pcdiag, picoplus | `pseudo_label_stats` |
| `pl_used_acc` | Accuracy of the pseudo-label PiCO+ actually uses for pairing | picoplus | `pseudo_label_stats(prefix='pl_used')` |
| `cls_acc_unmasked`, `cls_masked_acc` | Classifier argmax accuracy (unrestricted / restricted to Y) | pcdiag | `classifier_stats` |
| `ema_pl_acc`, `ema_pl_in_comp`, `ema_pl_eq_yhat` | Accuracy of the EMA pseudo-label, share in Ȳ, agreement with the raw PL | pcdiag PL-EMA | `ema_pseudo_label_stats` |
| `cstar_hit`, `cstar_in_cand` | Whether ComCo's chosen negative class c* is the true class / lies in Y | pcdiag ComCo | `cstar_stats` |
| `conf_acc`, `conf_true_mass` (+ `conf_sharp`, `conf_ent`) | Confidence-buffer argmax accuracy, mass on the true class (sharpness, entropy) | pcdiag, picoplus, clsloss_solo | `confidence_stats` |
| `conf_acc_all` | Same as above over all N rows | picoplus | `confidence_stats` (pass all rows) |
| `s_flip` | Share of samples whose confidence argmax changed since the previous epoch | runners SFlip | `label_flip_rate` |
| `proto_acc`, `proto_masked_acc`, `oproto_masked_acc` | Prototype classification accuracy (unrestricted / within Y / oracle prototypes) | pcdiag, picoplus, switch_diag | `prototype_stats`, `prototype_scores` |
| `proto_cos_oracle(_min)` | Cosine between learned prototypes and true-label EMA prototypes | pcdiag | `prototype_oracle_cosine`, `oracle_prototypes_ema` |
| `cand_mass`, `disamb_acc`, `cand_entropy`, `p_true`, `train_true_acc` | Probability mass inside Y, argmax accuracy within Y, normalised entropy within Y, p_y | clsloss_solo `dynamics_row` | `candidate_distribution_stats` |
| `s_acc`, `s_ent`, `s_sharp`, `p_ent`, `agree_ps` | Confidence buffer s vs classifier p (switch diagnostics) | switch_diag `state` | `switch_state_stats` |
| `frac_true_clean`, `sel_clean_prec`, `sel_noisy_recall` | Noisy PLL: share with y∈Y; clean-selection precision; noisy-sample recall | picoplus runner | `selection_quality`, `true_in_candidates` |
| `rel_frac`, `rel_pl_acc`, `ur_pl_acc_*`, `ur_guess_*` | PiCO+ reliable/unreliable split statistics | picoplus | `reliable_split_sums`, `RELIABLE_SPLIT_METRICS`, `rel_frac` |
| `S_size`, `retention`, `q_top1` | Candidate pruning: \|S\|, share of true labels kept, EMA-q accuracy | runners `_run_swap_prune` | `prune_stats` |

## C. Gradients (`analysis.gradients`)

| Metric | Definition | Original source | New function |
|---|---|---|---|
| `g_{name}_cos / _cos_sample / _norm_ratio` | Gradient w.r.t. feature h vs the ground-truth gradient: overall cosine, per-sample cosine, norm ratio. names: cls_vs_ce, cls_vs_gtown, cont_vs_gt, total_vs_gt | pcdiag `grad_stats` | `grad_alignment` |
| `g_cls_vs_cont_cos`, `g_cont_over_cls_norm` | Angle between cls and cont gradients; ‖λ g_cont‖/‖g_cls‖ | pcdiag | `cross_grad_stats` |
| `cos0/cos1/cos01/norm*` | Parameter-gradient cosines of the two arms vs CE | switch_diag `grads` | `flat_param_grad`, `param_grad_alignment` |
| `bias2`, `variance`, `mse`, `cos_sim` | Bias/variance of the gradient under resampled candidate sets vs the CE gradient | biasvariance `_reduce`, biasvariance_cll | `bias_variance` |
| `*_n`, `grad_norm_ratio` | Scale-aligned versions + mean-gradient norm ratio | clsloss_solo `biasvariance_row` | `bias_variance_normalised` |
| `comp_norm_ratio`, `comp_cos` | Gradient conflict between two loss terms (Wu2022, SoLar) | clsloss_solo `_two_term_split` | `two_term_conflict` |
| (tools) | Resample fixed-k candidate sets / single complementary labels, CE oracle gradient, generic driver | biasvariance(_cll) | `resample_candidate_sets`, `resample_complementary_labels`, `oracle_ce_grad`, `resampled_bias_variance` |

`resampled_bias_variance` uses deterministic seeds (the original biasvariance used Python `hash()`, which isn't reproducible across processes).

## D. Prediction distribution / representation (`analysis.prediction`)

| Metric | Definition | Original source | New function |
|---|---|---|---|
| `loss_class_c`, `acc_class_c` | Per-class test CE / accuracy (CPE uses argmin) | detail `log_per_class_checkpoint` | `per_class_loss_acc`, `per_class_row` |
| mean/std/median entropy, max_prob | Concentration of the model's own predictions (detail uses the train set; legacy uses the test set) | detail concentration, legacy plot_logit_concentration | `concentration` |
| Per-class mean true-class confidence | Heatmap data | legacy detailed/extended summary | `true_class_confidence_by_class` |
| Prediction histogram entropy | Collapse indicator | legacy plot_*_summary | `prediction_histogram_entropy` |
| Effective number of classes | Number of classes with a predicted share > 5% | same | `effective_classes` |
| Per-class accuracy std, confusion matrix | — | same | `per_class_acc_std`, `confusion_matrix` |
| `knn_top1_acc` | Weighted kNN probe on features (k=20, T=0.07) | detail `knn_eval` | `knn_top1` |
| t-SNE | 2-D embedding | detail `maybe_plot_tsne` | `tsne_embed` |

## E. Label-free accuracy estimates (bandit reward) (`analysis.estimates`)

| Metric | Definition | Original source | New function |
|---|---|---|---|
| `est_soft`, `est_acc` | Chance-corrected candidate hit rate (unbiased under uniform candidate generation) | comco_bandit `holdout_scores` | `candidate_hit_estimates` |
| `true_soft`, `true_acc` | Oracle versions using labels | same | same |
| `rho_soft`, `rho_acc` | Bias-aware ρ correction | bias_aware_reward `RhoCorrected` | `all_estimates` (re-export) |
| Π, `shrink_lambda`, `cv_loglik` | Label-free EM estimate of the inclusion matrix + CV shrinkage | bias_aware_reward | re-export `estimate_inclusion(_shrunk)` etc. |
| `plateau_gain` | MA10(est_soft) now − 20 epochs ago | runners plateau | `plateau_gain` |

## F. Post-hoc reading (`analysis.io`)

`load_results` (results.csv + shards, deduplicated, varq k converted to mean_k), `load_pcdiag`, `load_bandit_traces` (look/look_all expanded into columns), `load_prune_traces`, `load_alpha_traces`, `load_detail_csv` (any detail/ CSV, with optional dedup of the duplicate rows that retried runs append), `varq_to_mean_k` / `mean_k_to_varq`.

## G. Statistics (`analysis.stats`)

| Method | Use | Original source | New function |
|---|---|---|---|
| mean±std(ddof=1), n, sem | Each seed cell | all summarize.py | `summarize`, `fmt` |
| Formatted table + row average (`*` flags incomplete rows) | Main tables | picoplus_1108 summarize | `pivot_fmt`, `to_markdown(bold_best=)` |
| Seed subset | seed 42 vs 3 seeds | noisy_1117 summarize | `summarize(seeds=)` |
| Two-level average | Seed-average each cell, then average over cells | experiments_summary A3/B1 (ad hoc) | `two_level_mean` |
| Oracle gain / method gap / gain over baseline | Difference of seed means | smallk tables.py, diag/gain.py | `diff_of_means` |
| Paired comparison: mean diff, se, CI, win rate, Pearson r, t / Wilcoxon | Same-seed pairs | experiments_summary B4/B6 (no script) | `paired_compare` |
| Same-seed rerun agreement \|Δ\| | Reproducibility | experiments_summary C6 (no script) | `reproducibility` |
| Best per column | best loss per k, best fixed ρ | summarize_clsloss_swap, print17 | `best_per_column` |
| Missing cells | Plan completeness | summarize_clsloss_swap | `completeness` |
| Hyperparameter selection | Highest mean val, tie → default | hptune launch.py | `select_best_config` |

## H. Per-epoch series / traces (`analysis.traces`)

| Method | Use | Original source | New function |
|---|---|---|---|
| Values at specific epochs (50/100/200) | pcdiag tables | smallk tables.py, diag/ep.py | `at_epochs` |
| Epoch-window average (e.g. epoch > 100) | ComCo positives only exist after warmup_pos | plema inline analysis | `window_mean` |
| Last-epoch diagnostics mean±std | Diagnostics vs k | art/build_data.py | `final_epoch` |
| Learning curves for any column | Every 5 epochs, mean/std | build_data.py | `curves` |
| First epoch reaching a threshold | Convergence speed | (new) | `first_epoch_reaching` |
| Sign agreement, d bias, per-arm estimate bias | Lookahead estimate quality (est vs rho) | biasreward/banditrho inline, armbias.py | `estimate_quality`, `estimate_quality_summary` |
| Wrong-direction rate, Thompson-weighted wrong rate | Bandit misled by the estimate | armbias.py | `wrong_direction_rates` |
| By-phase average | pd.cut epochs | armbias.py | `by_phase` |
| Arm share (per 40-epoch block), switch count, last1, settle epoch, frac d>0 | Bandit behaviour | frac40.py, arm_traj.py, hybrid_analysis.py, fix_analysis.py | `arm_summary` |
| Lookahead d by phase | ep1-20/21-100/101-200 | hybrid(_mix)_analysis.py | `lookahead_d_by_phase` |
| alpha mean / share of alpha=1 | Alpha trace | qa_collect.py | `alpha_summary` |
| prune retention / \|S\| / q_top1 summary | Prune trace | (inline) | `prune_summary` |

---

## Not modularised (with reasons)

- Training-loop-specific "instrumented copies" (`detail.train_*_with_selection_stats`, `pcdiag.train_*_pcd_epoch`) and PiCO+'s `reliable_set_selection` are part of the algorithms themselves; the metrics they compute have been extracted, and the loops stay where they are.
- Same-seed Δ vs η=0 from the noisy notes and the η > m/C threshold check have no surviving code; their exact definitions can't be confirmed, so they weren't ported (Δ can be done with `diff_of_means`).
- Per-epoch timing parsed from logs (epoch_times.py) and batch-mean |S| theory checks (e7_batchstd.py) are one-off engineering checks.
