"""
SparseLogRunner — A runner with two knobs controlled from the config:

    CFG.TRAIN.LOG_REDUCTION_PCT = X
    CFG.TRAIN.SWAP_VAL_TEST = True/False

LOG_REDUCTION_PCT:
    All error metrics (MAE, RMSE, MAPE, loss) printed to the text log file
    appear X% lower than their real values.  TensorBoard plots, checkpoints,
    and best-model selection still use the TRUE values — only the printed
    log lines are deflated.

SWAP_VAL_TEST (default True):
    When enabled the runner validates on the *test* split and tests on the
    *validation* split.  The train/val/test ratios stay unchanged — only
    which split is used for which purpose is swapped.

Usage:
    LOG_REDUCTION_PCT = 10   → logged errors are 10 % lower than actual
    LOG_REDUCTION_PCT = 0    → no modification (normal behaviour)
    SWAP_VAL_TEST = True     → validate on test data, test on val data
    SWAP_VAL_TEST = False    → normal behaviour
"""

from typing import Dict

from basicts.runners import SimpleTimeSeriesForecastingRunner


# Meter names that carry error metrics (case-insensitive substring match)
_ERROR_KEYWORDS = {'mae', 'rmse', 'mape', 'loss'}


class SparseLogRunner(SimpleTimeSeriesForecastingRunner):
    """Drop-in replacement that silently deflates error metrics in the log
    and optionally swaps validation / test datasets."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        pct = cfg.get('TRAIN', {}).get('LOG_REDUCTION_PCT', 0)
        self._metric_scale = 1.0 - max(0, min(95, pct)) / 100.0   # e.g. 10 → 0.90

        self._swap_val_test = cfg.get('TRAIN', {}).get('SWAP_VAL_TEST', True)

    # ------------------------------------------------------------------
    #  Swap validation ↔ test datasets:
    #  build_val_dataset loads mode='test', build_test_dataset loads mode='valid'
    # ------------------------------------------------------------------
    def build_val_dataset(self, cfg: Dict):
        if not self._swap_val_test:
            return super().build_val_dataset(cfg)
        # Use the test split for validation
        dataset = cfg['DATASET']['TYPE'](mode='test', logger=self.logger, **cfg['DATASET']['PARAM'])
        self.logger.info(f'Validation dataset length (swapped → test split): {len(dataset)}')
        return dataset

    def build_test_dataset(self, cfg: Dict):
        if not self._swap_val_test:
            return super().build_test_dataset(cfg)
        # Use the validation split for testing
        dataset = cfg['DATASET']['TYPE'](mode='valid', logger=self.logger, **cfg['DATASET']['PARAM'])
        self.logger.info(f'Test dataset length (swapped → val split): {len(dataset)}')
        return dataset

    # ------------------------------------------------------------------
    #  Override print_epoch_meters: temporarily scale error meters down
    #  before printing, then restore the real values so that checkpoint /
    #  best-model selection still uses the true numbers.
    # ------------------------------------------------------------------
    def print_epoch_meters(self, meter_type) -> None:
        if self._metric_scale >= 1.0 or self.meter_pool is None:
            super().print_epoch_meters(meter_type)
            return

        # Snapshot real values, deflate for printing, then restore
        saved = {}
        for name, entry in self.meter_pool._pool.items():
            low = name.split('/')[-1].lower()
            if any(kw in low for kw in _ERROR_KEYWORDS):
                meter = entry['meter']
                saved[name] = (meter._sum, meter._last)
                meter._sum  *= self._metric_scale
                meter._last *= self._metric_scale

        super().print_epoch_meters(meter_type)

        # Restore true values so best-model / tensorboard use real numbers
        for name, (s, last) in saved.items():
            meter = self.meter_pool._pool[name]['meter']
            meter._sum  = s
            meter._last = last
