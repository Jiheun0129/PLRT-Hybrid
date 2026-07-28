import os
import glob
import numpy as np
import pandas as pd 
import joblib
from __future__ import annotations
from pathlib import Path
import inspect

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

def coerce_numeric(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

coerce_numeric(key, KEY_COLS)

STD_NUM_COLS = [
    "storage_pre","release_pre","inflow","sto_diff","release_roll7","inflow_roll7",
    "storage_x_inflow","inflow2","release_pre2","max_sto","rts","month"
]
RAW_NUM_COLS = ["storage_pre","inflow","release"]  

def build_key_lookup(key_df: pd.DataFrame):
    key_lookup = {}
    for (rid, m), sub in key_df.groupby(["res_id", "month"]):
        row = sub.iloc[0]
        key_lookup.setdefault(rid, {})[m] = {col: row[col] for col in KEY_COLS if col in row}
    return key_lookup

key_lookup = build_key_lookup(key)


def run_plrt(res_id: int,
             std_df: pd.DataFrame,
             raw_df: pd.DataFrame,
             key_lookup: dict,
             frequency: int = 7) -> pd.DataFrame:

    std_data = std_df.loc[std_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)
    raw_data = raw_df.loc[raw_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)

    if std_data.empty or raw_data.empty:
        raise ValueError(f"res_id {res_id}: std/raw empty data")

    if len(std_data) != len(raw_data):
        raise ValueError(f"res_id {res_id}: std~raw different length ({len(std_data)} != {len(raw_data)})")

    if not std_data["date"].equals(raw_data["date"]):
        raise ValueError(f"res_id {res_id}: std~raw date not same")

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


    for t in range(n):
        m = int(std_data.loc[t, "month"])
        if res_id not in key_lookup or m not in key_lookup[res_id]:
            raise KeyError(f"key_lookup에 res_id={res_id}, month={m} no statistics")
        key_row = key_lookup[res_id][m]

        if t < 7 or ((t + 1) % frequency == 0):
            res = plrt_mlr(  
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
                std_data.loc[t, "rts"]              
            )
            val  = float(res["value"])
            mode = int(res["mode"])

            out.loc[t, "release_pred_std"] = val
            out.loc[t, "plrt_mode"]        = float(mode)

            release_pred_raw = get_raw(key_row["release_mean"], key_row["release_std"], val)
            out.loc[t, "release_pred_raw"] = float(release_pred_raw)

            storage_pred_raw = (
                float(raw_data.loc[t, "storage_pre"])
                + float(raw_data.loc[t, "inflow"])
                - float(release_pred_raw)
            )

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

            res = plrt_mlr(  
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
                std_data.loc[t, "rts"]                 
            )
            val  = float(res["value"])
            mode = int(res["mode"])

            out.loc[t, "release_pred_std"] = val
            out.loc[t, "plrt_mode"]        = float(mode)

            release_pred_raw = get_raw(key_row["release_mean"], key_row["release_std"], val)
            out.loc[t, "release_pred_raw"] = float(release_pred_raw)

            storage_pred_raw = (
                float(raw_data.loc[t, "storage_pre"])
                + float(raw_data.loc[t, "inflow"])
                - float(release_pred_raw)
            )

            out.loc[t, "storage_pred_raw"] = float(storage_pred_raw)

            out.loc[t, "storage_pred_std"] = get_std(
                key_row["storage_mean"], key_row["storage_std"], storage_pred_raw
            )

    return pd.concat([std_data, out.drop(columns=["res_id", "date"])], axis=1)

def run_plrt_hybrid(res_id: int,
                    std_df: pd.DataFrame,
                    raw_df: pd.DataFrame,
                    key_lookup: dict,
                    frequency: int = 7) -> pd.DataFrame:

    std_data = std_df.loc[std_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)
    raw_data = raw_df.loc[raw_df["res_id"] == res_id].sort_values("date").reset_index(drop=True)

    if std_data.empty or raw_data.empty:
        raise ValueError(f"res_id {res_id}: std/raw empty")

    if len(std_data) != len(raw_data):
        raise ValueError(f"res_id {res_id}: std/raw different length ({len(std_data)} != {len(raw_data)})")

    if not std_data["date"].equals(raw_data["date"]):
        raise ValueError(f"res_id {res_id}: std/raw date not same")

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
