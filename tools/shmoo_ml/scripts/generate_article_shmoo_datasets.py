"""
Generate 7 Specialized Shmoo Plot Datasets based on Design & Reuse Article:
"Understanding Shmoo Plots and Various Terminology of Testers"

Generates:
1. Normal_Shmoo_Dataset_500.csv (Well-behaved linear Fmax)
2. Brick_Wall_Shmoo_Dataset_500.csv (Bi-stable uninitialized register stripes)
3. Wall_Shmoo_Dataset_500.csv (Low-voltage IR drop wall + High-voltage hold time wall)
4. Reverse_Speedpath_Shmoo_Dataset_500.csv (Inverted curve: leakage at high VDD)
5. Floor_Shmoo_Dataset_500.csv (Low-frequency dynamic node leakage failure)
6. Finger_Shmoo_Dataset_500.csv (Crosstalk / resonance harmonic notch fingers)
7. Marginality_HoldTime_Shmoo_Dataset_500.csv (Supply rail droop & clock skew)
"""

import os
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


# 1. Normal Shmoo (Well-Behaved Linear Speedpath)
def gen_normal():
    df = make_grid()
    # Linear Fmax: Fmax = 3.2 * V - 1.4 (at 0.70V -> 0.84 GHz, at 1.20V -> 2.44 GHz)
    fmax = 3.2 * df['VDD_V'] - 1.4
    pass_mask = df['Frequency_GHz'] <= fmax
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(pass_mask, 'NA', 'FREQ_MARGIN')
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'] + np.random.uniform(-0.02, 0.02, len(df)), 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 120 + df['Frequency_GHz'] * 45, 1)
    df['Temperature_C'] = 25.0
    
    out_file = OUT_DIR / "Normal_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


# 2. Brick Wall Shmoo (Bi-stable Initialization Issue)
def gen_brick_wall():
    df = make_grid()
    fmax = 3.2 * df['VDD_V'] - 1.4
    base_pass = df['Frequency_GHz'] <= fmax
    
    # Bi-stable reset fails in alternating / vertical patterns across test columns
    vdd_indices = {v: i for i, v in enumerate(vdd_range)}
    is_failing_col = df['VDD_V'].map(lambda v: (vdd_indices[v] % 4 == 1 or vdd_indices[v] % 4 == 3))
    brick_fail = is_failing_col & (df['Point_ID'] % 2 == 0)
    
    pass_mask = base_pass & (~brick_fail)
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(
        pass_mask, 'NA',
        np.where(brick_fail, 'BRICK_WALL_INIT', 'FREQ_MARGIN')
    )
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'], 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 120 + df['Frequency_GHz'] * 45, 1)
    df['Temperature_C'] = 25.0
    
    out_file = OUT_DIR / "Brick_Wall_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


# 3. Wall Shmoo (Low Voltage Gate Drive Wall + High Voltage Hold Time Wall)
def gen_wall():
    df = make_grid()
    fmax = 3.2 * df['VDD_V'] - 1.4
    
    low_wall = df['VDD_V'] < 0.85      # Fails below 0.85V regardless of frequency
    high_wall = df['VDD_V'] > 1.12     # Fast paths cause hold-time race failure above 1.12V
    freq_fail = df['Frequency_GHz'] > fmax
    
    pass_mask = (~low_wall) & (~high_wall) & (~freq_fail)
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(
        pass_mask, 'NA',
        np.where(low_wall, 'LOW_VDD_WALL',
        np.where(high_wall, 'HOLD_TIME_WALL', 'FREQ_MARGIN'))
    )
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'], 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 120 + df['Frequency_GHz'] * 45, 1)
    df['Temperature_C'] = 25.0
    
    out_file = OUT_DIR / "Wall_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


# 4. Reverse Speedpath Shmoo (Dynamic Node Leakage at High VDD)
def gen_reverse_speedpath():
    df = make_grid()
    # Rises from 0.70V to 0.92V, then degrades downward due to leakage & RC delay
    v = df['VDD_V']
    # Parabolic / downward inverted speedpath: peak at 0.92V = 2.05 GHz, drops to 1.10 GHz at 1.20V
    fmax = np.where(v <= 0.92, 0.80 + (v - 0.70) * 5.68, 2.05 - (v - 0.92) * 3.4)
    
    pass_mask = df['Frequency_GHz'] <= fmax
    high_leak_fail = (~pass_mask) & (df['VDD_V'] > 0.92)
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(
        pass_mask, 'NA',
        np.where(high_leak_fail, 'REVERSE_SPEEDPATH_LEAKAGE', 'FREQ_MARGIN')
    )
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'], 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 150 + df['Frequency_GHz'] * 60, 1)  # Higher leakage current
    df['Temperature_C'] = 85.0  # Elevated temperature accentuates leakage
    
    out_file = OUT_DIR / "Reverse_Speedpath_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


# 5. Floor Shmoo (Dynamic Node Leakage at Low Frequencies)
def gen_floor():
    df = make_grid()
    fmax = 3.2 * df['VDD_V'] - 1.4
    
    # Fails at low frequencies (F < 1.10 GHz) because clock cycle is too slow and dynamic node discharges
    floor_fail = df['Frequency_GHz'] < 1.10
    freq_fail = df['Frequency_GHz'] > fmax
    
    pass_mask = (~floor_fail) & (~freq_fail)
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(
        pass_mask, 'NA',
        np.where(floor_fail, 'FLOOR_LEAKAGE_FAIL', 'FREQ_MARGIN')
    )
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'], 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 120 + df['Frequency_GHz'] * 45, 1)
    df['Temperature_C'] = 75.0
    
    out_file = OUT_DIR / "Floor_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


# 6. Finger Shmoo (Crosstalk / Resonance Harmonic Notch Fingers)
def gen_finger():
    df = make_grid()
    fmax = 3.2 * df['VDD_V'] - 1.4
    
    # Resonant coupling notches at [1.25, 1.45] GHz and [1.85, 1.98] GHz
    notch_1 = (df['Frequency_GHz'] >= 1.25) & (df['Frequency_GHz'] <= 1.45) & (df['VDD_V'] >= 0.85)
    notch_2 = (df['Frequency_GHz'] >= 1.85) & (df['Frequency_GHz'] <= 1.98) & (df['VDD_V'] >= 0.95)
    finger_fail = notch_1 | notch_2
    
    freq_fail = df['Frequency_GHz'] > fmax
    pass_mask = (~finger_fail) & (~freq_fail)
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(
        pass_mask, 'NA',
        np.where(finger_fail, 'FINGER_RESONANCE_COUPLING', 'FREQ_MARGIN')
    )
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'], 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 120 + df['Frequency_GHz'] * 45, 1)
    df['Temperature_C'] = 25.0
    
    out_file = OUT_DIR / "Finger_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


# 7. Marginality & Hold Time Shmoo
def gen_marginality():
    df = make_grid()
    fmax = 3.2 * df['VDD_V'] - 1.4
    
    ir_drop_fail = (df['VDD_V'] < 0.82) & (df['Frequency_GHz'] > 0.90)
    skew_timing = (df['VDD_V'] > 1.08) & (df['Frequency_GHz'] > 2.00)
    freq_fail = df['Frequency_GHz'] > fmax
    
    pass_mask = (~ir_drop_fail) & (~skew_timing) & (~freq_fail)
    
    df['Test_Result'] = np.where(pass_mask, 'PASS', 'FAIL')
    df['Failure_Code'] = np.where(
        pass_mask, 'NA',
        np.where(ir_drop_fail, 'MARGINALITY_IR_DROP',
        np.where(skew_timing, 'TIMING', 'FREQ_MARGIN'))
    )
    df['Margin_GHz'] = np.round(fmax - df['Frequency_GHz'], 3)
    df['Timing_ns'] = np.round(1.0 / df['Frequency_GHz'], 3)
    df['Current_mA'] = np.round(df['VDD_V'] * 130 + df['Frequency_GHz'] * 50, 1)
    df['Temperature_C'] = 25.0
    
    out_file = OUT_DIR / "Marginality_HoldTime_Shmoo_Dataset_500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated: {out_file.name} (Pass: {(df['Test_Result']=='PASS').sum()}/{len(df)})")


if __name__ == '__main__':
    print("=== Generating 7 Design & Reuse Shmoo Datasets (500 checkpoints each) ===")
    gen_normal()
    gen_brick_wall()
    gen_wall()
    gen_reverse_speedpath()
    gen_floor()
    gen_finger()
    gen_marginality()
    print("=== All Datasets Successfully Generated in shmoo_dataset/ ===")
