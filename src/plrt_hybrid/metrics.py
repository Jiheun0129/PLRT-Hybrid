import os
import glob
import numpy as np
import pandas as pd 
import joblib
from __future__ import annotations
from pathlib import Path
import inspect

def get_r2(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)

    if a.size < 10:
        return np.nan
    if np.isnan(a).any() or np.isnan(p).any():
        return np.nan

    ss_res = np.sum((a - p) ** 2)
    mean_a = np.mean(a)
    ss_tot = np.sum((a - mean_a) ** 2)

    if ss_tot == 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)

