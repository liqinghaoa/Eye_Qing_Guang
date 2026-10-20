"""Explicit training precision, with unchanged interpretation of old configs."""


def training_precision(settings):
    precision = settings.get("precision", "fp16" if settings["amp"] else "fp32")
    if not isinstance(precision, str) or precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("training.precision must be fp32, fp16 or bf16.")
    if type(settings["amp"]) is not bool or settings["amp"] != (precision != "fp32"):
        raise ValueError("training.amp must be false for fp32 and true for fp16/bf16.")
    return precision
