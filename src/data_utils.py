# This code has been adapted from: https://github.com/harrylui1995/ASP_E2EPO , the official GitHub Repo for Lui, G. N., & Demirel, S. (2025). Gradient-based smart predict-then-optimize framework for aircraft arrival scheduling problem. Journal of Open Aviation Science, 2(2). https://doi.org/10.59490/joas.2024.7891

import ast
import numpy as np
import pandas as pd

def safe_eval_list(s):
    try:
        if isinstance(s, str):
            evaluated = ast.literal_eval(s)
            if not isinstance(evaluated, (list, tuple)):
                return None
            return evaluated
        elif isinstance(s, (list, tuple)):
            return s
        return None
    except Exception:
        return None

def load_and_clean_csv(csv_path, n_rows=500):
    df = pd.read_csv(csv_path).head(n_rows)

    list_columns = [
        'costs', 'wtc', 'original_feats', 'T_mean', 'T', 'callsigns',
        'transit_times', 'relative_transit_times', 'cost_transit_time_diff',
        'feats', 'transit_time_difference'
    ]

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_eval_list)

            # float-cast most list cols (except callsigns, wtc)
            if col not in ['callsigns', 'wtc']:
                df[col] = df[col].apply(
                    lambda x: [float(i) if isinstance(i, (int, float, str)) else i for i in x]
                    if isinstance(x, list) else x
                )
    return df

def build_feature_matrix(df, feature_col='original_feats'):
    feats_lengths = []
    feats_list = []

    for _, row in df.iterrows():
        flattened = [item for sublist in row[feature_col] for item in sublist]
        combined = flattened  # keep minimal (no extra features)
        feats_lengths.append(len(combined))

    lengths = pd.Series(feats_lengths, index=df.index)
    most_common_length = lengths.mode()[0]
    df = df.drop(lengths[lengths != most_common_length].index)

    for _, row in df.iterrows():
        flattened = [item for sublist in row[feature_col] for item in sublist]
        feats_list.append(flattened)

    X = np.array(feats_list)
    return df, X
