"""Models (2026-11-22 refactor).

    backbones  ResNet18 / MLP single-encoder classifiers (was src/models.py)
    common     shared dual-encoder pieces: key-encoder momentum, prototype EMA, predict_logits
The dual-encoder model classes still live in src/pico/model.py, src/comco/model.py,
src/pcdiag.py, src/picoplus.py, src/hybrid_switch.py, src/solo_switch.py.
"""

from src.models.backbones import (MLPClassifier, ResNet18, create_mlp_model, create_model,  # noqa: F401
                                  create_model_for_spec)
