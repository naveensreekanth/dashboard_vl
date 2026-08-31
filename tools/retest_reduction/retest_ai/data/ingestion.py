import io
import os
import pandas as pd
from typing import Dict, Optional, Union, Any
from ..config.settings import DATA_FILES, ALL_MODEL_FEATURES, IDENTIFIER_COLS

def load_dataset(filepath_or_key: str) -> pd.DataFrame:
    """
    Load an Excel dataset from a file path or known key ('month_0', 'month_6', 'month_12', 'ai_dataset').
    """
    if filepath_or_key in DATA_FILES:
        path = DATA_FILES[filepath_or_key]
    else:
        path = filepath_or_key

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at path: {path}")

    # Read first sheet
    df = pd.read_excel(path, sheet_name=0)
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    return df


def load_pre_retest_workbook(source: Any) -> pd.DataFrame:
    """
    Load a pre-retest event workbook from a path or file-like object.
    Does not interpret Ground_Truth / outcome columns as prediction input.
    """
    if hasattr(source, "getvalue"):
        excel_source = io.BytesIO(source.getvalue())
    elif hasattr(source, "read"):
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        excel_source = source
    else:
        excel_source = source
    df = pd.read_excel(excel_source, sheet_name=0)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Loads Month 0, Month 6, Month 12, and AI Baseline datasets.
    """
    datasets = {}
    for key, path in DATA_FILES.items():
        if os.path.exists(path):
            datasets[key] = load_dataset(path)
        else:
            print(f"Warning: File {path} for {key} does not exist.")
    return datasets
