import os
import glob
import numpy as np
import pandas as pd 
import joblib
from __future__ import annotations
from pathlib import Path
import inspect


d
def _safe_predict(model, X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    try:
        sig = inspect.signature(model.predict)
        if "verbose" in sig.parameters:  # Keras
            y = model.predict(X, verbose=0)
        else:  # scikit-learn 
            y = model.predict(X)
    except (TypeError, ValueError, AttributeError):
        y = model.predict(X)

    return float(np.asarray(y).squeeze()) if X.shape[0] == 1 else np.asarray(y).squeeze()

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def get_std(mean, stdev, value):
    mean  = _as_float(mean)
    stdev = _as_float(stdev)
    value = _as_float(value)

    if np.isnan(mean):  mean  = 0.0
    if np.isnan(stdev) or stdev == 0: 
        return value
    return (value - mean) / stdev

def get_raw(mean, stdev, value):
    mean  = _as_float(mean)
    stdev = _as_float(stdev)
    value = _as_float(value)

    if np.isnan(mean):  mean  = 0.0
    if np.isnan(stdev) or stdev == 0: 
        return value + mean  

    return stdev * value + mean