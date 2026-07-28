import os
import glob
import numpy as np
import pandas as pd 
import joblib
from __future__ import annotations
from pathlib import Path
import inspect


MODEL_FILES = {
    "modes_3_4_5": "M345.pkl",
    "modes_7_8": "M78.pkl",
    "mode_9": "M9.pkl",
    "mode_10": "M10.pkl",
    "modes_11_12": "M1112.pkl",
    "mode_13": "M13.pkl",
    "modes_16_17": "M16M17.pkl",
}


def load_models(model_directory: str | Path) -> dict:
    """Load pretrained MLP models used by PLRT-Hybrid."""

    model_directory = Path(model_directory)
    models = {}

    for name, filename in MODEL_FILES.items():
        model_path = model_directory / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing pretrained model: {model_path}"
            )

        models[name] = joblib.load(model_path)

    return models



def _mlp_predict1(model, feats14) -> float:
    return _safe_predict(model, np.asarray(feats14, dtype=np.float32))

def _mlp_predict2(model, feats14) -> float:
    return _safe_predict(model, np.asarray(feats14, dtype=np.float32))

def _mlp_predict3(model, feats9) -> float:
    return _safe_predict(model, np.asarray(feats9, dtype=np.float32))

def _mlp_predict4(model, feats14) -> float:
    return _safe_predict(model, np.asarray(feats14, dtype=np.float32))

def _mlp_predict5(model, feats14) -> float:
    return _safe_predict(model, np.asarray(feats14, dtype=np.float32))

def _mlp_predict6(model, feats9) -> float:
    return _safe_predict(model, np.asarray(feats9, dtype=np.float32))

def _mlp_predict7(model, feats9) -> float:
    return _safe_predict(model, np.asarray(feats9, dtype=np.float32))

def plrt_mlp(storage_pre: float, release_pre: float, inflow: float, sto_diff: float,
             release_roll7: float, inflow_roll7: float, storage_x_inflow: float,
             inflow2: float, release_pre2: float, max_sto: float, rts: float, is_weekend: float, 
             month: float, sin_DOY: float, cos_DOY: float, quarter: float) -> dict:

    vals = [storage_pre, release_pre, inflow, sto_diff,
            release_roll7, inflow_roll7, storage_x_inflow,
            inflow2, release_pre2, max_sto, rts, is_weekend, month, sin_DOY, cos_DOY, quarter]
    if any(pd.isna(v) for v in vals):
        raise ValueError("❌ MISSING INPUT in plrt_mlp()")

    feats9 = [storage_pre, release_pre, inflow, sto_diff,
              release_roll7, inflow_roll7, storage_x_inflow,
              inflow2, release_pre2]
    
    feats14 = [storage_pre, release_pre, inflow, sto_diff,
              release_roll7, inflow_roll7, storage_x_inflow,
              inflow2, release_pre2, is_weekend, month, sin_DOY, cos_DOY, quarter]

    if max_sto <= 185505416.716:  # Mode 1~6
        if max_sto <= 8339264.432:  # Mode 1
            val = (0.000 + storage_pre*0.033 + release_pre*0.454 +
                   inflow*0.352 + sto_diff*0.102 + release_roll7*0.287 +
                   inflow_roll7*(-0.204) + storage_x_inflow*0.038 +
                   inflow2*0.237 + release_pre2*(-0.150))
            return {"value": val, "mode": 1}

        else:
            if rts <= 189.225:  # Mode 2~5
                if rts <= 13.971:  # Mode 2
                    val = (0.000 + storage_pre*0.081 + release_pre*0.445 +
                           inflow*0.291 + sto_diff*0.081 + release_roll7*0.175 +
                           inflow_roll7*(-0.105) + storage_x_inflow*0.154 +
                           inflow2*0.141 + release_pre2*(-0.091))
                    return {"value": val, "mode": 2}

                else:  # Mode 3~5
                    if release_pre <= 0.726:
                        if storage_x_inflow <= 0.152:  # Mode 3
                            val = _mlp_predict1(mlp_1, feats14)
                            return {"value": val, "mode": 3}
                        else:  # Mode 4 (MLP)
                            val = _mlp_predict1(mlp_1, feats14)
                            return {"value": val, "mode": 4}
                    else:  # Mode 5
                        val = _mlp_predict1(mlp_1, feats14)
                        return {"value": val, "mode": 5}

            else:  # Mode 6
                val = (0.000 + storage_pre*0.007 + release_pre*0.999 +
                       inflow*(-0.002) + sto_diff*0.063 + release_roll7*0.004 +
                       inflow_roll7*(-0.029) + storage_x_inflow*0.051 +
                       inflow2*0.040 + release_pre2*(-0.091))
                return {"value": val, "mode": 6}

    else:  # Mode 7~17
        if rts <= 23.848:  # Mode 7,8
            if max_sto <= 795061466.513:  # Mode 7
                val = _mlp_predict2(mlp_2, feats14)
                return {"value": val, "mode": 7}
            else:  # Mode 8
                val = _mlp_predict2(mlp_2, feats14)
                return {"value": val, "mode": 8}

        else:  # Mode 9~17
            if release_pre <= -0.561:  # Mode 9~12
                if max_sto <= 959064570.924:  # Mode 9,10
                    if rts <= 158.356:  # Mode 9 (MLP)
                        val = _mlp_predict3(mlp_3, feats9)
                        return {"value": val, "mode": 9}
                    else:  # Mode 10 (MLP)
                        val = _mlp_predict4(mlp_4, feats14)
                        return {"value": val, "mode": 10}
                else:  # Mode 11,12
                    if inflow2 <= -0.134:  # Mode 11 (MLP)
                        val = _mlp_predict5(mlp_5, feats14)
                        return {"value": val, "mode": 11}
                    else:  # Mode 12 (MLP)
                        val = _mlp_predict5(mlp_5, feats14)
                        return {"value": val, "mode": 12}

            else:  # Mode 13~17
                if storage_pre <= 0.852:  # Mode 13~15
                    if inflow <= -0.594:  # Mode 13
                        val =  _mlp_predict6(mlp_6, feats9)
                        return {"value": val, "mode": 13}
                    else:
                        if max_sto <= 1510090693.013:  # Mode 14
                            val = (-0.013 + storage_pre*0.014 + release_pre*0.824 +
                                   inflow*0.004 + sto_diff*0.356 + release_roll7*0.084 +
                                   inflow_roll7*0.018 + storage_x_inflow*0.031 +
                                   inflow2*(-0.006) + release_pre2*(-0.011))
                            return {"value": val, "mode": 14}
                        else:  # Mode 15
                            val = (-0.035 + storage_pre*0.045 + release_pre*0.654 +
                                   inflow*0.120 + sto_diff*0.485 + release_roll7*0.195 +
                                   inflow_roll7*0.034 + storage_x_inflow*(-0.136) +
                                   inflow2*(-0.040) + release_pre2*0.041)
                            return {"value": val, "mode": 15}

                else:  # Mode 16,17
                    if release_roll7 <= 1.030:  # Mode 16
                        val = _mlp_predict7(mlp_7, feats9)
                        return {"value": val, "mode": 16}
                    else:  # Mode 17
                        val = _mlp_predict7(mlp_7, feats9)
                        return {"value": val, "mode": 17}
