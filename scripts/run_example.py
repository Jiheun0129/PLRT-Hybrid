import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from __future__ import annotations
import joblib
import inspect
from __future__ import annotations

#########################################################################################################
#Import Data
meta = pd.read_csv("data/example/41_meta.csv")
key = pd.read_csv("data/example/41_key.csv")
std  = "data/example/41_std.csv"
raw  = "data/example/41_raw.csv"

mlp_1 = joblib.load("/models/M345.pkl")
mlp_2  = joblib.load("/models/M78.pkl")
mlp_3  = joblib.load("/models/M9.pkl")
mlp_4  = joblib.load("/models/M10.pkl")
mlp_5  = joblib.load("/models/M1112.pkl")
mlp_6  = joblib.load("/models/M13.pkl")
mlp_7  = joblib.load("/models/M1617.pkl")

#########################################################################################################
# Basic Functions
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

def coerce_numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

coerce_numeric(key, KEY_COLS)

def build_key_lookup(key_df: pd.DataFrame):
    """
    return: key_lookup[res_id][month] = {col: value, ...}
    """
    key_lookup = {}
    for (rid, m), sub in key_df.groupby(["res_id", "month"]):
        row = sub.iloc[0]
        key_lookup.setdefault(rid, {})[m] = {col: row[col] for col in KEY_COLS if col in row}
    return key_lookup


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

#########################################################################################################
#PLRT_Hybrid Functions
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

def run_plrt_hybrid(res_id: int,
                    std_df: pd.DataFrame,
                    raw_df: pd.DataFrame,
                    key_lookup: dict,
                    frequency: int = 7) -> pd.DataFrame:

    std_data = std_df.loc[std_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)
    raw_data = raw_df.loc[raw_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)

    n = len(std_data)

    out = std_data[["res_id", "date"]].copy()
    cols_init = [
        "release_pred_raw","release_pred_std","storage_pred_raw","storage_pred_std",
        "release_pred_roll7_raw","release_pred_pre_raw","release_pred_pre2_raw",
        "storage_pred_roll7_raw","storage_pred_pre_raw","sto_diff_pred_raw","storage_x_inflow_pred_raw",
        "release_pred_roll7_std","release_pred_pre_std","release_pred_pre2_std",
        "storage_pred_roll7_std","storage_pred_pre_std","sto_diff_pred_std","storage_x_inflow_pred_std",
        "plrt_mode"
    ]
    for c in cols_init:
        out[c] = np.nan

    for c in cols_init:
        out[c] = out[c].astype("float64")

    release_max_global = float(raw_data["release"].max(skipna=True))*1.1 if "release" in raw_data.columns else np.inf

    for t in range(n):
        m = int(std_data.loc[t, "month"])
        if res_id not in key_lookup or m not in key_lookup[res_id]:
            raise KeyError(f"key_lookup에 res_id={res_id}, month={m} 통계가 없음")
        key_row = key_lookup[res_id][m]

        if t < 7 or ((t + 1) % frequency == 0):
            res = plrt_mlp(  
                std_data.loc[t, "storage_pre"],      
                std_data.loc[t, "release_pre"],   
                std_data.loc[t, "inflow"],           
                std_data.loc[t, "sto_diff"],        
                std_data.loc[t, "release_roll7"],    
                std_data.loc[t, "inflow_roll7"],      
                std_data.loc[t, "storage_x_inflow"],  
                std_data.loc[t, "inflow2"],       
                std_data.loc[t, "release_pre2"],      
                std_data.loc[t, "max_sto"],       
                std_data.loc[t, "rts"],            
                std_data.loc[t, "is_weekend"],  
                std_data.loc[t, "month"],          
                std_data.loc[t, "sin_DOY"],      
                std_data.loc[t, "cos_DOY"],          
                std_data.loc[t, "quarter"]
            )
            val  = float(res["value"])
            mode = int(res["mode"])

            out.loc[t, "release_pred_std"] = val
            out.loc[t, "plrt_mode"]        = float(mode)

            release_pred_raw = get_raw(key_row["release_mean"], key_row["release_std"], val)
            release_pred_raw = max(0.0, min(release_pred_raw, release_max_global))
            out.loc[t, "release_pred_raw"] = float(release_pred_raw)

            storage_pred_raw = (
                float(raw_data.loc[t, "storage_pre"])
                + float(raw_data.loc[t, "inflow"])
                - float(release_pred_raw)
            )
            storage_pred_raw = max(0.0, min(storage_pred_raw, float(std_data.loc[t, "max_sto"]) * 0.9))
            out.loc[t, "storage_pred_raw"] = float(storage_pred_raw)

            out.loc[t, "storage_pred_std"] = get_std(
                key_row["storage_mean"], key_row["storage_std"], storage_pred_raw
            )

        else:
            r7r = float(np.mean(out.loc[t-7:t-1, "release_pred_raw"]))
            s7r = float(np.mean(out.loc[t-7:t-1, "storage_pred_raw"]))

            rpr = float(out.loc[t-1, "release_pred_raw"])
            rpr2 = rpr ** 2
            spr = float(out.loc[t-1, "storage_pred_raw"])
            st_diff = spr - s7r
            sxi = spr * float(raw_data.loc[t, "inflow"])

            out.loc[t, "release_pred_roll7_raw"]    = r7r
            out.loc[t, "storage_pred_roll7_raw"]    = s7r
            out.loc[t, "release_pred_pre_raw"]      = rpr
            out.loc[t, "release_pred_pre2_raw"]     = rpr2
            out.loc[t, "storage_pred_pre_raw"]      = spr
            out.loc[t, "sto_diff_pred_raw"]         = st_diff
            out.loc[t, "storage_x_inflow_pred_raw"] = sxi

            out.loc[t, "release_pred_roll7_std"] = get_std(
                key_row["release_roll7_mean"], key_row["release_roll7_std"], r7r
            )
            out.loc[t, "storage_pred_roll7_std"] = get_std(
                key_row["storage_roll7_mean"], key_row["storage_roll7_std"], s7r
            )
            out.loc[t, "release_pred_pre_std"] = get_std(
                key_row["release_pre_mean"], key_row["release_pre_std"], rpr
            )
            out.loc[t, "release_pred_pre2_std"] = get_std(
                key_row["release_pre2_mean"], key_row["release_pre2_std"], rpr2
            )
            out.loc[t, "storage_pred_pre_std"] = get_std(
                key_row["storage_pre_mean"], key_row["storage_pre_std"], spr
            )
            out.loc[t, "sto_diff_pred_std"] = get_std(
                key_row["sto_diff_mean"], key_row["sto_diff_std"], st_diff
            )
            out.loc[t, "storage_x_inflow_pred_std"] = get_std(
                key_row["storage_x_inflow_mean"], key_row["storage_x_inflow_std"], sxi
            )

            res = plrt_mlp( 
                out.loc[t, "storage_pred_pre_std"],   
                out.loc[t, "release_pred_pre_std"],   
                std_data.loc[t, "inflow"],            
                out.loc[t, "sto_diff_pred_std"],      
                out.loc[t, "release_pred_roll7_std"],   
                std_data.loc[t, "inflow_roll7"],        
                out.loc[t, "storage_x_inflow_pred_std"],
                std_data.loc[t, "inflow2"],          
                out.loc[t, "release_pred_pre2_std"],   
                std_data.loc[t, "max_sto"],             
                std_data.loc[t, "rts"],                  
                std_data.loc[t, "is_weekend"],  
                std_data.loc[t, "month"],         
                std_data.loc[t, "sin_DOY"],     
                std_data.loc[t, "cos_DOY"],         
                std_data.loc[t, "quarter"]
            )
            val  = float(res["value"])
            mode = int(res["mode"])

            out.loc[t, "release_pred_std"] = val
            out.loc[t, "plrt_mode"]        = float(mode)

            release_pred_raw = get_raw(key_row["release_mean"], key_row["release_std"], val)
            release_pred_raw = max(0.0, min(release_pred_raw, release_max_global))
            out.loc[t, "release_pred_raw"] = float(release_pred_raw)

            storage_pred_raw = (
                float(raw_data.loc[t, "storage_pre"])
                + float(raw_data.loc[t, "inflow"])
                - float(release_pred_raw)
            )
            storage_pred_raw = max(0.0, min(storage_pred_raw, float(std_data.loc[t, "max_sto"]) * 0.9))
            out.loc[t, "storage_pred_raw"] = float(storage_pred_raw)

            out.loc[t, "storage_pred_std"] = get_std(
                key_row["storage_mean"], key_row["storage_std"], storage_pred_raw
            )

    return pd.concat([std_data, out.drop(columns=["res_id", "date"])], axis=1)


#########################################################################################################
# Run Example

KEY_COLS = [
    "release_mean","release_std",
    "storage_mean","storage_std",
    "release_roll7_mean","release_roll7_std",
    "storage_roll7_mean","storage_roll7_std",
    "release_pre_mean","release_pre_std",
    "release_pre2_mean","release_pre2_std",
    "storage_pre_mean","storage_pre_std",
    "sto_diff_mean","sto_diff_std",
    "storage_x_inflow_mean","storage_x_inflow_std",
]

STD_NUM_COLS = [
    "storage_pre","release_pre","inflow","sto_diff","release_roll7","inflow_roll7",
    "storage_x_inflow","inflow2","release_pre2","max_sto","rts","month"
]
RAW_NUM_COLS = ["storage_pre","inflow","release"]

key_lookup = build_key_lookup(key)

FREQS    = [1, 2, 7, 30, 90, 180]
METHODS  = {
    "plrt_hybrid": run_plrt_hybrid, 
}

for df in (std, raw, key):
    if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "month" not in df.columns and "date" in df.columns:
        df["month"] = df["date"].dt.month

key_lookup = build_key_lookup(key)

pred_df = run_plrt_hybrid(rid, std_df=std, raw_df=raw, key_lookup=key_lookup, frequency=2) #Change Initialization frequency here

obs = raw.loc[raw["res_id"] == rid].sort_values("date")["release"].values
pred = pred_df["release_pred_raw"].values
r2 = get_r2(obs, pred)
print("R^2 =", r2)
