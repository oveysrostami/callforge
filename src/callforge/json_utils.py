"""Strict JSON at persistence and HTTP boundaries, including legacy ASR metrics."""
import json
import math


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def dumps(value, **kwargs):
    kwargs["allow_nan"] = False
    return json.dumps(finite_json(value), **kwargs)
