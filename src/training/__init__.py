__all__ = ["NERTrainer", "NERMetrics", "NERMetricComputer"]


def __getattr__(name: str):
    if name == "NERTrainer":
        from .trainer import NERTrainer
        return NERTrainer
    if name in ("NERMetrics", "NERMetricComputer"):
        from .evaluator import NERMetrics, NERMetricComputer
        return NERMetrics if name == "NERMetrics" else NERMetricComputer
    raise AttributeError(f"module 'src.training' has no attribute {name!r}")
