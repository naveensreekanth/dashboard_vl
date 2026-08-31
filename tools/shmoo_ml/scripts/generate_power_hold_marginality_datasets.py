"""
Generate Dedicated Shmoo Plot Datasets for:
1. Marginality Issue Shmoo (VDDL/VDDH corner failures)
2. Power & IR Drop Issue Shmoo (High dynamic switching current rail collapse)
3. Hold Time Issue Shmoo (Fast-path logic race condition & clock skew at high VDD)
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "shmoo_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_VDD = 20
N_FREQ = 25  # 20 * 25 = 500 checkpoints per dataset

vdd_range = np.round(np.linspace(0.70, 1.20, N_VDD), 3)
freq_range = np.round(np.linspace(0.60, 2.50, N_FREQ), 3)


def make_grid():
    rows = []
    pid = 1
    for v in vdd_range:
        for f in freq_range:
            rows.append({'Point_ID': pid, 'Die_ID': 'D0001', 'VDD_V': float(v), 'Frequency_GHz': float(f)})
            pid += 1
    return pd.DataFrame(rows)


# 1. Marginality Issue Shmoo Dataset
def gen_marginality_issue():
    df = make_grid()
    fmax = 3.2 * df['VDD_V'] - 1.4
    
    # Fails at low voltage rail VDDL (< 0.82V) at high frequencies (setup violation)
    v_low_fail = (df['VDD_V'] <= 0.82) & (df['Frequency_GHz'] > 0.85)
    # Fails at high voltage rail VDDH (> 1.10V) at high frequencies (PLL jitter / rail droop)
    v_high_fail = (df['VDD_V'] >= 1.10) & (df['Frequency_GHz'] > 1.95)
    freq_fail = df['Frequency_GHz'] > fmax
    
    pass_mask = (~v_low_fail) & (~v_high_fail) & (~freq_fail)
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(
        pass_mask, 'NA',
        np.where(v_low_fail, 'MARGINALITY_VDDL_FAIL',
        np.where(v_high_fail, 'MARGINALITY_VDDH_FAIL', 'FREQ_MARGIN'))
    )
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'], 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 125 + df['Frequency_GHz'] * 48, 1)
    df['Temperature_C'] = 25.0
    
    out_file = OUT_DIR / "Marginality_Issue_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


# 2. Power & IR Drop Issue Shmoo Dataset
def gen_power_irdrop_issue():
    df = make_grid()
    fmax = 3.2 * df['VDD_V'] - 1.4
    
    # Dynamic power P = C * V^2 * f causes internal IR drop on supply rail.
    # At high frequencies (F > 1.65 GHz), dynamic switching current collapses internal VDD.
    power_droop_fail = df['Frequency_GHz'] > 1.65
    freq_fail = df['Frequency_GHz'] > fmax
    
    pass_mask = (~power_droop_fail) & (~freq_fail)
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(
        pass_mask, 'NA',
        np.where(power_droop_fail, 'POWER_IR_DROP_FAIL', 'FREQ_MARGIN')
    )
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'], 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 220 + df['Frequency_GHz'] * 110, 1)  # High switching current
    df['Temperature_C'] = 85.0  # High thermal dissipation
    
    out_file = OUT_DIR / "Power_Issue_IRDrop_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


# 3. Hold Time Race Issue Shmoo Dataset
def gen_hold_time_issue():
    df = make_grid()
    fmax = 3.2 * df['VDD_V'] - 1.4
    
    # Hold time violation occurs at elevated VDD (VDD > 1.05V) irrespective of frequency
    hold_violation = df['VDD_V'] > 1.05
    freq_fail = df['Frequency_GHz'] > fmax
    
    pass_mask = (~hold_violation) & (~freq_fail)
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(
        pass_mask, 'NA',
        np.where(hold_violation, 'HOLD_TIME_VIOLATION', 'FREQ_MARGIN')
    )
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'], 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 130 + df['Frequency_GHz'] * 50, 1)
    df['Temperature_C'] = 25.0
    
    out_file = OUT_DIR / "Hold_Time_Race_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


if __name__ == '__main__':
    print("=== Generating Dedicated Power, Hold Time, and Marginality Shmoo Datasets (500 checkpoints each) ===")
    gen_marginality_issue()
    gen_power_irdrop_issue()
    gen_hold_time_issue()
    print("=== Datasets Successfully Generated in shmoo_dataset/ ===")
