"""Factories for the score models used by the command-line entry points."""

from .rank import FactorModel
from .temporal import MODEL_ID, TemporalScorer

FACTORS = ("trend", "macd")
ARTIFACT_LOADERS = {MODEL_ID: TemporalScorer}


def load_models(strategy, artifact_paths, source, device):
    models = [FactorModel(key) for key in FACTORS]
    for key, path in artifact_paths.items():
        if key not in ARTIFACT_LOADERS:
            raise ValueError(f"未注册的训练模型 {key}。")
        if strategy.weights.get(key, 0) > 0:
            models.append(ARTIFACT_LOADERS[key](path, source, device))
    return models
