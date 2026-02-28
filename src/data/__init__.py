# Lazy imports to avoid requiring all dependencies at import time.
# Import individual modules directly (e.g. from src.data.annotator import BIOAnnotator).

__all__ = [
    "NVDFetcher",
    "CVEPreprocessor",
    "BIOAnnotator",
    "DataAugmentor",
    "NERDataset",
    "NERDataLoader",
]


def __getattr__(name: str):
    if name == "NVDFetcher":
        from .nvd_fetcher import NVDFetcher
        return NVDFetcher
    if name == "CVEPreprocessor":
        from .preprocessor import CVEPreprocessor
        return CVEPreprocessor
    if name == "BIOAnnotator":
        from .annotator import BIOAnnotator
        return BIOAnnotator
    if name == "DataAugmentor":
        from .augmentor import DataAugmentor
        return DataAugmentor
    if name in ("NERDataset", "NERDataLoader"):
        from .dataset import NERDataset, NERDataLoader
        return NERDataset if name == "NERDataset" else NERDataLoader
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
