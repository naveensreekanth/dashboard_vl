import os
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def extract_docx_text(docx_path):
    print(f"=== EXTRACTING DOCX: {docx_path} ===")
    if not os.path.exists(docx_path):
        print(f"File not found: {docx_path}")
        return ""
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            for p in tree.iterfind('.//w:p', namespaces):
                texts = [node.text for node in p.iterfind('.//w:t', namespaces) if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            full_text = '\n'.join(paragraphs)
            print(f"Extracted {len(paragraphs)} paragraphs, {len(full_text)} chars.")
            return full_text
    except Exception as e:
        print(f"Error reading {docx_path}: {e}")
        return ""

def inspect_excel(filepath):
    print(f"\n=======================================================")
    print(f"INSPECTING WORKBOOK: {os.path.basename(filepath)}")
    print(f"=======================================================")
    if not os.path.exists(filepath):
        print("File does not exist!")
        return None
    
    xl = pd.ExcelFile(filepath)
    print(f"Sheet names: {xl.sheet_names}")
    
    df = pd.read_excel(filepath, sheet_name=0)
    print(f"Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        null_cnt = df[col].isnull().sum()
        unique_cnt = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:3]
        print(f"  {i+1:2d}. {col:30s} | Type: {str(dtype):10s} | Nulls: {null_cnt:3d} | Unique: {unique_cnt:4d} | Samples: {sample_vals}")
    
    print("\nHead (First 3 rows):")
    print(df.head(3).to_string())
    
    # Key column distributions if present
    for col in ['Ground_Truth', 'Retest_Result', 'Final_Result', 'AI_Recommendation', 'Fail_Test', 'ATE_Site', 'Wafer_ID', 'Test_Month']:
        if col in df.columns:
            print(f"\nValue counts for '{col}':")
            print(df[col].value_counts(dropna=False).to_string())
            
    # Numerical summaries
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print("\nNumerical columns summary:")
        print(df[num_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].to_string())
        
    return df

def run_phase1_inspection():
    print("STARTING PHASE 1 INSPECTION...\n")
    
    # 1. DOCX Inspection
    docx_files = ['RETEST~2.docx', 'AI Recommended Retest report .docx']
    for d in docx_files:
        text = extract_docx_text(d)
        if text:
            print("\n--- SAMPLE TEXT (first 2500 chars) ---")
            print(text[:2500])
            print("\n--- TEXT MATCHING KPI / METRICS / ACCURACY ---")
            for line in text.split('\n'):
                if any(kw in line.lower() for kw in ['accuracy', 'precision', 'recall', 'specificity', 'unnecessary', 'missed', 'kpi', 'threshold', '30%', '70.4', '69.9', '82.9', '853']):
                    print("  -> ", line)

    # 2. Excel Datasets
    dfs = {}
    files = [
        'ATE_Retest_50_Devices_Month_0_Historical.xlsx',
        'ATE_Retest_50_Devices_Month_6_Historical.xlsx',
        'ATE_Retest_50_Devices_Month_12_NEW_Inference.xlsx',
        'ATE_Retest_50_Devices_AI_Dataset.xlsx'
    ]
    for f in files:
        if os.path.exists(f):
            dfs[f] = inspect_excel(f)
            
    # 3. Longitudinal / Device / Event tracking consistency check
    print("\n=======================================================")
    print("LONGITUDINAL DEVICE & EVENT TRACKING CHECK")
    print("=======================================================")
    m0_file = 'ATE_Retest_50_Devices_Month_0_Historical.xlsx'
    m6_file = 'ATE_Retest_50_Devices_Month_6_Historical.xlsx'
    m12_file = 'ATE_Retest_50_Devices_Month_12_NEW_Inference.xlsx'
    
    if m0_file in dfs and m6_file in dfs and m12_file in dfs:
        df0 = dfs[m0_file]
        df6 = dfs[m6_file]
        df12 = dfs[m12_file]
        
        dev0 = set(df0['Device_ID']) if 'Device_ID' in df0.columns else set()
        dev6 = set(df6['Device_ID']) if 'Device_ID' in df6.columns else set()
        dev12 = set(df12['Device_ID']) if 'Device_ID' in df12.columns else set()
        
        print(f"Device counts -> Month 0: {len(dev0)}, Month 6: {len(dev6)}, Month 12: {len(dev12)}")
        print(f"Intersection Month 0 & Month 6: {len(dev0.intersection(dev6))}")
        print(f"Intersection Month 0, 6, 12: {len(dev0.intersection(dev6).intersection(dev12))}")
        
        if 'Failure_Event' in df0.columns and 'Failure_Event' in df6.columns and 'Failure_Event' in df12.columns:
            ev0 = set(df0['Failure_Event'])
            ev6 = set(df6['Failure_Event'])
            ev12 = set(df12['Failure_Event'])
            print(f"Failure_Event counts -> Month 0: {len(ev0)}, Month 6: {len(ev6)}, Month 12: {len(ev12)}")
            print(f"Exact match across months for events? {ev0 == ev6 == ev12}")
            
        print("\nSchema Comparison across Months:")
        all_cols = sorted(list(set(df0.columns).union(set(df6.columns)).union(set(df12.columns))))
        schema_comp = []
        for c in all_cols:
            schema_comp.append({
                'Column': c,
                'Month_0': c in df0.columns,
                'Month_6': c in df6.columns,
                'Month_12_Inference': c in df12.columns,
                'Type_M0': str(df0[c].dtype) if c in df0.columns else '-',
                'Type_M6': str(df6[c].dtype) if c in df6.columns else '-',
                'Type_M12': str(df12[c].dtype) if c in df12.columns else '-'
            })
        print(pd.DataFrame(schema_comp).to_string())

if __name__ == '__main__':
    run_phase1_inspection()
