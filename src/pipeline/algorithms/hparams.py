"""Per-algorithm default hyperparameters.

Optimizer/lr/wd choices mirror the values already validated in
scripts/run_adam_comparison.py, scripts/run_solar_comparison.py, etc.:
Adam lr=3e-4 for most algorithms; SGD lr=0.01 momentum=0.9 for PRODEN and
SoLar, which follow their original papers.
"""

_ADAM = dict(optimizer='adam', lr=3e-4, weight_decay=1e-4)
_SGD = dict(optimizer='sgd', lr=0.01, momentum=0.9, weight_decay=1e-4)

ALGO_HPARAMS = {
    'CLPL':     _ADAM,
    'Wu2022':   _ADAM,
    'PRODEN':   _SGD,
    'MCL-LOG':  _ADAM,
    'MCL-LOG-Fixed': _ADAM,
    'SCL-NL':   _ADAM,
    'OP':       _ADAM,
    'OP-W':     _ADAM,
    'CPE':      _ADAM,
    'PiCO':     _ADAM,
    'PiCO-Fixed': _ADAM,
    'PiCO-MCL': _ADAM,
    'PiCO-SC':  _ADAM,
    'PiCO-CLS': _ADAM,
    'ComCo':    _ADAM,
    'ComCo-Fixed': _ADAM,
    'SoLar':    _SGD,
}


def make_optimizer(model, hparams: dict):
    import torch.optim as optim

    if hparams['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    return optim.SGD(model.parameters(), lr=hparams['lr'],
                      momentum=hparams.get('momentum', 0.9), weight_decay=hparams['weight_decay'])
