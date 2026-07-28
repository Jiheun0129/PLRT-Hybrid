import os
import glob
import numpy as np
import pandas as pd 
import joblib
from __future__ import annotations
from pathlib import Path
import inspect

def plrt_mlr(storage_pre: float, release_pre: float, inflow: float, sto_diff: float,
             release_roll7: float, inflow_roll7: float, storage_x_inflow: float,
             inflow2: float, release_pre2: float, max_sto: float, rts: float) -> dict:

    vals = [storage_pre, release_pre, inflow, sto_diff,
            release_roll7, inflow_roll7, storage_x_inflow,
            inflow2, release_pre2, max_sto, rts]
    if any(pd.isna(v) for v in vals):
        raise ValueError("❌ MISSING INPUT in plrt_mlr()")

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
                            val = (-0.008 + storage_pre*0.018 + release_pre*0.835 +
                                   inflow*0.084 + sto_diff*0.090 + release_roll7*0.020 +
                                   inflow_roll7*(-0.033) + storage_x_inflow*(-0.040) +
                                   inflow2*0.006 + release_pre2*0.125)
                            return {"value": val, "mode": 3}
                        else:  # Mode 4
                            val = (-0.080 + storage_pre*0.023 + release_pre*0.975 +
                                   inflow*0.202 + sto_diff*0.189 + release_roll7*0.068 +
                                   inflow_roll7*(-0.121) + storage_x_inflow*(0.101) +
                                   inflow2*(-0.071) + release_pre2*(-0.147))
                            return {"value": val, "mode": 4}
                    else:  # Mode 5
                        val = (0.157 + storage_pre*0.113 + release_pre*0.725 +
                               inflow*0.278 + sto_diff*0.179 + release_roll7*0.065 +
                               inflow_roll7*(-0.119) + storage_x_inflow*(-0.017) +
                               inflow2*0.002 + release_pre2*(-0.029))
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
                val = (0.000 + storage_pre*0.040 + release_pre*0.226 +
                       inflow*0.203 + sto_diff*0.050 + release_roll7*0.086 +
                       inflow_roll7*(-0.048) + storage_x_inflow*0.521 +
                       inflow2*(-0.007) + release_pre2*0.015)
                return {"value": val, "mode": 7}
            else:  # Mode 8
                val = (0.000 + storage_pre*0.080 + release_pre*0.526 +
                       inflow*0.456 + sto_diff*0.156 + release_roll7*0.132 +
                       inflow_roll7*(-0.080) + storage_x_inflow*(-0.069) +
                       inflow2*(-0.053) + release_pre2*0.056)
                return {"value": val, "mode": 8}

        else:  # Mode 9~17
            if release_pre <= -0.561:  # Mode 9~12
                if max_sto <= 959064570.924:  # Mode 9,10
                    if rts <= 158.356:  # Mode 9 
                       val = (-0.041 + storage_pre*0.025 + release_pre*0.734 +
                       inflow*0.041 + sto_diff*0.155 + release_roll7*0.131 +
                       inflow_roll7*(-0.001) + storage_x_inflow*(-0.019) +
                       inflow2*(-0.024) + release_pre2*0.057)
                       return {"value": val, "mode": 9}
                    else:  # Mode 10
                        val = (-0.018 + storage_pre*0.027 + release_pre*0.499 +
                        inflow*0.013 + sto_diff*0.712 + release_roll7*0.311 +
                        inflow_roll7*(-0.042) + storage_x_inflow*(0.013) +
                        inflow2*(-0.001) + release_pre2*0.150)
                        return {"value": val, "mode": 10}
                else:  # Mode 11,12
                    if inflow2 <= -0.134:  # Mode 11 
                        val = (-0.199 + storage_pre*0.035 + release_pre*0.380 +
                        inflow*0.067 + sto_diff*0.823 + release_roll7*0.370 +
                        inflow_roll7*(-0.036) + storage_x_inflow*(-0.085) +
                        inflow2*(-0.011) + release_pre2*(-0.050))
                        return {"value": val, "mode": 11}
                    else:  # Mode 12 
                        val = (-0.034 + storage_pre*0.049 + release_pre*0.113 +
                        inflow*(-0.018) + sto_diff*0.082 + release_roll7*0.402 +
                        inflow_roll7*(-0.013) + storage_x_inflow*(0.009) +
                        inflow2*(0.004) + release_pre2*0.394)
                        return {"value": val, "mode": 12}

            else:  # Mode 13~17
                if storage_pre <= 0.852:  # Mode 13~15
                    if inflow <= -0.594:  # Mode 13
                        val = (-0.062 + storage_pre*0.022 + release_pre*0.491 +
                               inflow*0.020 + sto_diff*0.460 + release_roll7*0.258 +
                               inflow_roll7*(-0.107) + storage_x_inflow*0.024 +
                               inflow2*0.033 + release_pre2*0.109)
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
                        val = (-0.137 + storage_pre*0.098 + release_pre*1.080 +
                               inflow*0.007 + sto_diff*0.433 + release_roll7*0.035 +
                               inflow_roll7*(-0.003) + storage_x_inflow*0.015 +
                               inflow2*0.022 + release_pre2*(-0.199))
                        return {"value": val, "mode": 16}
                    else:  # Mode 17
                        val = (-0.002 + storage_pre*0.094 + release_pre*0.953 +
                               inflow*0.087 + sto_diff*0.154 + release_roll7*0.003 +
                               inflow_roll7*(-0.025) + storage_x_inflow*(-0.033) +
                               inflow2*0.005 + release_pre2*(-0.057))
                        return {"value": val, "mode": 17}
