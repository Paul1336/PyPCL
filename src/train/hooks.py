"""Per-batch diagnostic hooks for src/train/{pico,comco}.py (2026-11-22 refactor).

They replace the instrumented copies of the training loops in pipeline/detail.py: the
statistic is computed with src.analysis.pairs and written with the SAME CSV writers / layout
(detail.log_*_stats_batch), so existing plots keep working.

Hook protocol: optional on_mask(ctx) (PiCO), on_masks(ctx) (ComCo), end_epoch(epoch).
"""

from src.analysis import pairs

NAN = float('nan')


class PiCOSelectionStats:
    """pico_selection_stats.csv: within-batch (q x q, self excluded) pair confusion per batch."""

    def __init__(self, raw_cfg, algorithm, C):
        self.raw_cfg, self.algorithm, self.C, self.rows = raw_cfg, algorithm, C, []

    def on_mask(self, ctx):
        mask = ctx['mask']
        if mask is None:
            self.rows.append((ctx['batch_idx'], NAN, NAN, 0, 0, 0, 0))
            return
        r = pairs.within_batch_confusion(mask, ctx['true'].to(ctx['device']))
        self.rows.append((ctx['batch_idx'], r['pos_precision'], r['neg_precision'], r['tp'], r['fp'], r['tn'], r['fn']))

    def end_epoch(self, epoch):
        from src.pipeline import detail
        detail.log_pico_selection_stats_batch(self.raw_cfg, self.algorithm, self.C, epoch + 1, self.rows)
        self.rows = []


class OracleCorrectionStats:
    """pico_oracle_correction_stats.csv (mode='remove') / pico_oracle_add_correction_stats.csv
    (mode='add'): precision before/after the graded oracle correction, negative precision of the
    corrected full-pool mask, counts."""

    def __init__(self, raw_cfg, algorithm, C, mode):
        self.raw_cfg, self.algorithm, self.C, self.mode, self.rows = raw_cfg, algorithm, C, mode, []

    def on_mask(self, ctx):
        s, mask = ctx['stats'], ctx['mask']
        neg = NAN if mask is None else pairs.negative_precision(mask, ctx['same_true_pool'])
        if self.mode == 'remove':
            self.rows.append((ctx['batch_idx'], s.get('precision_before', NAN), s.get('precision_after', NAN), neg,
                              s.get('pos_total', 0), s.get('n_flipped', 0)))
        else:
            self.rows.append((ctx['batch_idx'], s.get('precision_before', NAN), s.get('precision_after', NAN), neg,
                              s.get('pos_total', 0), s.get('n_added', 0), s.get('n_capped', False),
                              s.get('add_ratio', NAN)))

    def end_epoch(self, epoch):
        from src.pipeline import detail
        log = (detail.log_pico_oracle_correction_stats_batch if self.mode == 'remove'
               else detail.log_pico_oracle_add_correction_stats_batch)
        log(self.raw_cfg, self.algorithm, self.C, epoch + 1, self.rows)
        self.rows = []


class ComCoSelectionStats:
    """comco_selection_stats.csv: precision of the selected positives over the q+k block."""

    def __init__(self, raw_cfg, algorithm, C):
        self.raw_cfg, self.algorithm, self.C, self.rows = raw_cfg, algorithm, C, []

    def on_masks(self, ctx):
        prec, total = pairs.pos_precision_2b(ctx['pos_mask'], ctx['true'].to(ctx['device']))
        self.rows.append((ctx['batch_idx'], prec, total))

    def end_epoch(self, epoch):
        from src.pipeline import detail
        detail.log_comco_selection_stats_batch(self.raw_cfg, self.algorithm, self.C, epoch + 1, self.rows)
        self.rows = []
