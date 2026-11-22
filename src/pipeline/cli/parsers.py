"""argparse builders for scripts/run_pipeline.py (split out 2026-11-22)."""

import argparse  # noqa: F401

from src.pipeline.algorithms import ALL_ALGORITHM_NAMES
from src.pipeline.datasets import ALL_DATASET_NAMES


def _add_run_parser(sub):
    p = sub.add_parser('run', help='Train algorithms and record results')
    p.add_argument('--run_name', required=True, help='Name of this experiment; results go to results/<run_name>/')
    p.add_argument('--algorithms', nargs='+', default=ALL_ALGORITHM_NAMES, choices=ALL_ALGORITHM_NAMES)
    p.add_argument('--dataset', default='cifar100-subset', choices=ALL_DATASET_NAMES,
                    help="Which dataset to train on. 'cifar100-subset' (default) is the original "
                         "C-class-subset sweep; other values train on a fixed-class-count dataset "
                         "(see docs/00_paper_alignment_guide.md) and ignore --c_values (forced to "
                         "the dataset's native class count).")
    p.add_argument('--c_values', nargs='+', type=int, default=[5, 20],
                    help='Only used for --dataset cifar100-subset; ignored otherwise.')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                    help='One or more random seeds to sweep per (C, k, algorithm) cell -- each '
                         'seed gets its own row in results.csv (dedup/resume key includes seed, '
                         'so partially-completed seed sweeps resume correctly). Use `report` to '
                         'see per-seed values plus the mean/std across seeds. Diagnostic '
                         'instrumentation (--detail/--tsne/--concentration/--knn_eval) only runs '
                         'for the first seed in this list, regardless of how many are swept -- see '
                         'src/pipeline/detail.py _seed_matches. Default: 3 seeds.')
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--log_dir', default='logs/cifar100_subset')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--num_gpus', type=int, default=1)
    p.add_argument('--algo', default=None,
                    help='Pin this worker to one algorithm (overrides gpu_id round-robin)')
    p.add_argument('--report_every', type=int, default=10, help='Print ETA every N epochs')
    p.add_argument('--only_c', type=int, default=None, help='Run only this C value')
    p.add_argument('--only_k', nargs='+', type=int, default=None,
                    help='Run only this k value, or a specific list of k values (e.g. --only_k 5 8 10) '
                         'instead of the default per-C k-schedule.')
    p.add_argument('--only_q', type=float, default=None,
                    help='Use variable (q-based) candidate-label generation instead of a fixed k: '
                         'each false label is independently included w.p. q (0-1). Replaces the '
                         'whole k-schedule with a single cell. Mutually exclusive with --only_k. '
                         'Only supported for --dataset cifar100-subset, mnist, fashion-mnist, '
                         'kmnist, and cifar10 so far (DatasetSpec.supports_q).')
    p.add_argument('--only_bimodal', nargs=3, default=None, metavar=('K_LO', 'K_HI', 'P_LO'),
                    help='Bimodal candidate-set sizes: each sample gets K_LO candidates w.p. P_LO, else '
                         'K_HI. Replaces the k-schedule with one cell (cifar100-subset only).')
    p.add_argument('--batching', choices=['random', 'bucketed'], default='random',
                    help="PiCO/ComCo loader batching: 'random' (plain shuffle) or 'bucketed' (each "
                         "batch holds one candidate-set size only; see SizeBucketBatchSampler).")
    p.add_argument('--holdout_frac', type=float, default=0.0,
                    help='Hold this fraction of the training set out of every training loader '
                         '(cifar100-subset only); exposed to runners as loaders["holdout"]. '
                         'Needed by the ComCo-Weighted-Bandit* algorithms (label-free reward).')
    p.add_argument('--cl_bias', choices=['uniform', 'set1', 'set2', 'set3'], default='uniform',
                    help='Biased complementary-label generation (ComCo / Gao & Zhang 2021 Sec 4.3): per '
                         'true class the other labels are split into 3 groups drawn as complementary '
                         'labels w.p. set1 .75/.24/.01, set2 .66/.24/.10, set3 .45/.30/.25 (fixed-k, '
                         'cifar100-subset only). Not recorded in results.csv -- use one run_name per level.')
    p.add_argument('--noise_eta', type=float, default=0.0,
                    help='Noisy PLL (PiCO+ Sec 5.1 eta = 1 - P(y in Y)): each sample w.p. eta gets a candidate set of '
                         'k FALSE labels (|Y| = k kept; fixed-k uniform / --cl_bias only). Not recorded in '
                         'results.csv -- use one run_name per eta.')
    p.add_argument('--switch_diag', action='store_true',
                    help='Generic bandit runners: also log per-epoch confidence/model/prototype/gradient '
                         'diagnostics (src/switch_diag.py) into bandit_trace/.')
    p.add_argument('--detail', action='store_true',
                    help='Enable per-class-per-epoch-checkpoint diagnostic logging (replaces '
                         'scripts/legacy/run_extended_analysis.py): adds an extra full test-set '
                         'eval every --detail_log_every epochs, written to '
                         'results/<run_name>/detail/<algorithm>/C{C}_k{k}/per_class_loss.csv. For '
                         'PiCO specifically, also logs per-batch contrastive positive/negative '
                         'pair-selection precision (plus the raw tp/fp/tn/fn pair counts behind it) '
                         'against ground truth to pico_selection_stats.csv '
                         '(every batch, not gated by --detail_log_every). '
                         'Adds real overhead; off by default.')
    p.add_argument('--detail_log_every', type=int, default=10,
                    help='Epoch cadence for --detail per-class checkpoint logging.')
    p.add_argument('--tsne', action='store_true',
                    help='Every --tsne_every epochs, save a t-SNE snapshot of the contrastive '
                         'projection-head representation (raw embeddings + rendered PNG) to '
                         'results/<run_name>/detail/<algorithm>/C{C}_k{k}/tsne/. Independent of '
                         '--detail. Only meaningful for dual-encoder models (PiCO family, ComCo '
                         'family) -- silently does nothing for other algorithms. Adds a CPU-bound '
                         't-SNE fit (seconds) each time; off by default.')
    p.add_argument('--tsne_every', type=int, default=50, help='Epoch cadence for --tsne snapshots.')
    p.add_argument('--tsne_max_points', type=int, default=2000,
                    help='Max test-set samples to embed per --tsne snapshot (t-SNE cost grows '
                         'roughly with n log n).')
    p.add_argument('--concentration', action='store_true',
                    help='Log per-sample and averaged prediction-concentration (entropy + '
                         'max-softmax-prob of the model\'s own predicted distribution over the '
                         'training set) every --concentration_log_every epochs, to '
                         'results/<run_name>/detail/<algorithm>/C{C}_k{k}/concentration_summary.csv '
                         'and concentration/ep{epoch}.npz. Independent of --detail. Works for the '
                         'PiCO family, PRODEN family, and ComCo. Adds an extra full train-set '
                         'forward pass each time it fires; off by default.')
    p.add_argument('--concentration_log_every', type=int, default=10,
                    help='Epoch cadence for --concentration logging.')
    p.add_argument('--knn_eval', action='store_true',
                    help='Once at the end of training, evaluate kNN top-1 accuracy using the '
                         'contrastive encoder (train set as reference bank, test set as queries), '
                         'written to results/<run_name>/detail/<algorithm>/C{C}_k{k}/knn_eval.csv. '
                         'Only meaningful for dual-encoder models (PiCO/ComCo family) -- no-op '
                         'otherwise; off by default.')
    p.add_argument('--knn_eval_k', type=int, default=20, help='Number of neighbors for --knn_eval.')
    p.add_argument('--knn_temperature', type=float, default=0.07,
                    help='Softmax temperature for the --knn_eval weighted vote.')
    p.add_argument('--biasvariance', action='store_true',
                    help='Every --biasvariance_log_every epochs, freeze the current model/loss '
                         'state and measure the cls (and, for PiCO/ComCo family, contrastive) '
                         'loss gradient Bias^2/Variance/MSE against M resampled candidate-label '
                         'draws on a fixed eval batch (SCL-NL paper Section 4 style), written to '
                         'results/<run_name>/detail/<algorithm>/C{C}_k{k}/biasvariance.csv. '
                         'Unlike other --detail diagnostics, this runs for every --seeds value '
                         '(not just one designated seed), since measuring variance across '
                         'independently-trained-per-seed models is the point.')
    p.add_argument('--biasvariance_log_every', type=int, default=20)
    p.add_argument('--biasvariance_m_resamples', type=int, default=50,
                    help='Number of counterfactual candidate-label resamples per checkpoint.')
    p.add_argument('--biasvariance_eval_size', type=int, default=256,
                    help='Fixed eval-batch size (first N samples of the cell\'s train set, reused '
                         'across every checkpoint/epoch for comparability).')


def _add_merge_parser(sub):
    p = sub.add_parser('merge', help='Merge per-worker CSV shards into results.csv')
    p.add_argument('--run', required=True, dest='run_name')


def _add_report_parser(sub):
    p = sub.add_parser('report', help='Print per-seed accuracy plus mean/std across seeds, '
                                       'one row per (C, k, algorithm)')
    p.add_argument('--runs', nargs='+', required=True, help='run_name(s) to merge into one report')
    p.add_argument('--algorithms', nargs='+', default=None, choices=ALL_ALGORITHM_NAMES)
    p.add_argument('--c_values', nargs='+', type=int, default=None)
    p.add_argument('--out', default=None, help='Optional: also write the table as CSV to this path')
    p.add_argument('--confusion', action='store_true',
                    help="Also report each cell's FINAL-epoch PiCO contrastive pair-selection confusion "
                         "matrix (tp/fp/tn/fn, summed over that epoch's batches, plus tp/(tp+fp) as "
                         "'precision_pct' -- requires `run --detail`). Unlike accuracy, this is NOT "
                         "averaged across seeds: --detail only logs for one seed per cell (the "
                         "diagnostics_seed), so these columns reflect that seed alone. Blank for any "
                         "(C, k, algorithm) with no pico_selection_stats.csv (non-PiCO algorithms, or a "
                         "log that never reached prot_start).")
    p.add_argument('--sort_by', default='default',
                    choices=['default', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision'],
                    help="Row order. 'default': C, then k, then algorithm name (alphabetical) -- the "
                         "original grouping. 'tp'/'fp'/'tn'/'fn': descending by that --confusion raw "
                         "count (requires --confusion; rows with no confusion data sort last). "
                         "'accuracy': descending by mean accuracy. 'precision': descending by "
                         "tp/(tp+fp) (requires --confusion).")


def _add_plot_parser(sub):
    p = sub.add_parser('plot', help='Draw accuracy-vs-k figures from merged results')
    p.add_argument('--runs', nargs='+', required=True, help='run_name(s) to merge into one plot')
    p.add_argument('--algorithms', nargs='+', default=None, choices=ALL_ALGORITHM_NAMES)
    p.add_argument('--c_values', nargs='+', type=int, default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--group_by', default='paradigm', choices=['paradigm', 'all-in-one', 'per-algorithm'])
    p.add_argument('--rename', nargs='+', default=None,
                    help="Override legend/title labels, e.g. --rename PiCO-Fixed=PiCO ComCo-Fixed=ComCo "
                         "(data selection still uses the real algorithm name; only the shown text changes)")


def _add_plot_weight_parser(sub):
    p = sub.add_parser('plot-weight',
                        help='Accuracy vs. true-class weight (W), for one fixed (C, k) -- the parametrized '
                             'biased-init sweep (PiCO-Fixed-Biased{Cand,All}-W* / PRODEN-Biased{Cand,All}-W*) '
                             'complement to `plot`\'s accuracy-vs-k view')
    p.add_argument('--runs', nargs='+', required=True, help='run_name(s) to merge into one plot')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--weights', nargs='+', type=float, required=True,
                    help='True-class weight values (percentages), x-axis category order, e.g. '
                         '--weights 5 6 8 10 20, or fractional --weights 4.5 5.2 6.6 8.3 10 20')
    p.add_argument('--series', nargs='+', required=True,
                    help="TEMPLATE:LABEL pairs, one line per entry, e.g. "
                         "--series 'PiCO-Fixed-BiasedCand-W{w}:PiCO-BiasedCand' "
                         "'PiCO-Fixed-BiasedAll-W{w}:PiCO-BiasedAll' -- TEMPLATE must contain literal '{w}', "
                         "filled with each --weights value as a 2-digit zero-padded tag.")
    p.add_argument('--baselines', nargs='+', default=None,
                    help='ALGORITHM:LABEL pairs drawn as horizontal dashed reference lines, e.g. '
                         '--baselines PiCO-Fixed:PiCO PRODEN:PRODEN ComCo-Fixed:ComCo -- pulled from the '
                         'same --runs, so pass a run that actually has them if the swept sweep\'s own '
                         'run_name doesn\'t (e.g. --runs biased_rand_sweep biased_weight_sweep)')
    p.add_argument('--no_zero_pad', action='store_true',
                    help="Format {w} as a plain int instead of a 2-digit zero-padded tag -- use this for "
                         "the BiasedRand family's Wf (e.g. --series "
                         "'PiCO-Fixed-BiasedRand-W10-Wf{w}:PiCO W10' --weights 5 8 10 12 15 --no_zero_pad), "
                         "since biased_rand_variant_name's 'Wf{wf}' suffix isn't zero-padded (unlike W).")
    p.add_argument('--x_prefix', default='W', help="Tick-label / axis-label prefix, e.g. 'Wf' for the "
                                                     "BiasedRand family's Wf sweep instead of W.")
    p.add_argument('--xlabel', default=None, help='Override the x-axis label text.')
    p.add_argument('--title', default=None)
    p.add_argument('--out', required=True)


def _add_detail_plot_parser(sub):
    p = sub.add_parser('detail-plot', help='Per-class accuracy/loss heatmap over epoch checkpoints '
                                            '(requires --detail during run); one or two algorithms side by side')
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--alg_l', required=True, help='Left (or only) algorithm')
    p.add_argument('--alg_r', default=None, help='Right algorithm, for a side-by-side comparison')
    p.add_argument('--alg_l_display', default=None, help='Override --alg_l\'s title text only')
    p.add_argument('--alg_r_display', default=None, help='Override --alg_r\'s title text only')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--acc_only', action='store_true', help='Omit the CE-loss row (accuracy heatmap only)')
    p.add_argument('--side_by_side', action='store_true',
                    help='Single algorithm only: accuracy (left) / CE loss (right) side by side, '
                         'instead of the default stacked (accuracy on top, loss below) layout')
    p.add_argument('--show_class_names', action='store_true')
    p.add_argument('--seed', type=int, default=42, help='Only used with --show_class_names')
    p.add_argument('--data_dir', default='./data', help='Only used with --show_class_names')


def _add_detail_plot_multi_parser(sub):
    p = sub.add_parser('detail-plot-multi',
                        help='Per-class accuracy/loss heatmap for N algorithms side by side '
                             '(requires --detail during run); generalizes detail-plot beyond two algorithms')
    p.add_argument('--entries', nargs='+', required=True,
                    help='RUN:ALG:LABEL triples, one per column, in order, e.g. '
                         '--entries new_main_c20_k19_pico:PiCO:PiCO thresholdoracle_c20_k19_t1:PiCO-Oracle:PiCO-oracle')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--acc_only', action='store_true', help='Omit the CE-loss row (accuracy heatmaps only)')
    p.add_argument('--show_class_names', action='store_true')
    p.add_argument('--seed', type=int, default=42, help='Only used with --show_class_names')
    p.add_argument('--data_dir', default='./data', help='Only used with --show_class_names')


def _add_detail_plot_pico_parser(sub):
    p = sub.add_parser('detail-plot-pico', help="PiCO's contrastive positive/negative pair-selection "
                                                 "precision vs. ground truth, over epochs (requires --detail)")
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--alg', default='PiCO', help="Which PiCO-family algorithm's log to read (default: PiCO)")
    p.add_argument('--display_name', default=None, help="Override --alg's title text only")
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_pico_confusion_parser(sub):
    p = sub.add_parser('detail-plot-pico-confusion',
                        help="PiCO's contrastive pair-selection confusion matrix (TP/TN/FP/FN, as a "
                             "% of within-batch pairs, with raw counts annotated) vs. training step, "
                             "for one (algorithm, C, k) (requires --detail)")
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--alg', default='PiCO-Fixed', help="Which PiCO-family algorithm's log to read "
                                                         "(default: PiCO-Fixed)")
    p.add_argument('--display_name', default=None, help="Override --alg's title text only")
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_pico_confusion_all_parser(sub):
    p = sub.add_parser('detail-plot-pico-confusion-all',
                        help='Batch version of detail-plot-pico-confusion: one PNG per (algorithm, C, k) '
                             'combination found under results/<run>/detail/, all written into one folder '
                             '(requires --detail)')
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--out_dir', required=True, help='Folder to write every PNG into (created if missing)')
    p.add_argument('--algorithms', nargs='+', default=None,
                    help='Restrict to these algorithms only (default: every algorithm found under detail/)')
    p.add_argument('--c_values', nargs='+', type=int, default=None,
                    help='Restrict to these C values only (default: every C found)')
    p.add_argument('--k_values', nargs='+', type=int, default=None,
                    help='Restrict to these k values only (default: every k found)')


def _add_detail_plot_pico_multik_parser(sub):
    p = sub.add_parser('detail-plot-pico-multik',
                        help="Overlay one algorithm's positive-pair selection precision across several "
                             "k values on one chart (requires --detail on all of them)")
    p.add_argument('--runs', nargs='+', required=True,
                    help='k=run_name pairs, e.g. --runs 19=new_main_c20_k19_pico_fixed 15=new_main_c20_k15_pico_fixed')
    p.add_argument('--alg', default='PiCO-Fixed', help="Which algorithm's log to read (default: PiCO-Fixed)")
    p.add_argument('--display_name', default=None, help="Override --alg's title text only")
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_concentration_parser(sub):
    p = sub.add_parser('detail-plot-concentration',
                        help='Overlay per-algorithm prediction-concentration trends (mean entropy + '
                             'max-softmax-prob vs epoch, N algorithms) (requires `run --concentration`)')
    p.add_argument('--entries', nargs='+', required=True,
                    help='RUN:ALG:LABEL triples, one per line, in order, e.g. '
                         '--entries 0820_ablation:PiCO-Fixed:PiCO-Fixed '
                         '0820_ablation:PiCO-Fixed-UniformInit:UniformInit')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_knn_parser(sub):
    p = sub.add_parser('detail-plot-knn',
                        help='Bar chart comparing final kNN top-1 accuracy across N algorithms '
                             '(requires `run --knn_eval`)')
    p.add_argument('--entries', nargs='+', required=True,
                    help='RUN:ALG:LABEL triples, one per bar, in order')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)


def _add_detail_plot_pico_oracle_parser(sub):
    p = sub.add_parser('detail-plot-pico-oracle',
                        help="PiCO-Oracle's graduated correction: natural (pre-correction) vs. "
                             "post-correction positive-pair precision, over epochs (requires --detail)")
    p.add_argument('--run', required=True, dest='run_name')
    p.add_argument('--hide_neg', action='store_true', help='Omit the negative-pair precision line')
    p.add_argument('--C', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--out', required=True)
