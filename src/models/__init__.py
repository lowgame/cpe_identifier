__all__ = ["BERTForNER", "XLNetForNER", "GPT2ForNER", "BaseNERModel"]


def __getattr__(name: str):
    if name == "BaseNERModel":
        from .base_model import BaseNERModel
        return BaseNERModel
    if name == "BERTForNER":
        from .bert_ner import BERTForNER
        return BERTForNER
    if name == "XLNetForNER":
        from .xlnet_ner import XLNetForNER
        return XLNetForNER
    if name == "GPT2ForNER":
        from .gpt2_ner import GPT2ForNER
        return GPT2ForNER
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
