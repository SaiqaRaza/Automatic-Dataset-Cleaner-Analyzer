# app.py
"""
Interactive Preprocessing Streamlit App
- Upload CSV / Excel
- Inspect dataset
- Choose handling for missing values, encoding, scaling, outliers
- Produce cleaned CSV, pipeline decisions (.pkl), HTML report, plots, ZIP download
Notes:
- For one-hot we use pandas.get_dummies (safe control over expansion)
- For high-cardinality we prefer frequency / hash / ordinal to avoid huge expansion
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import zipfile
import json
import hashlib
from datetime import datetime
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

st.set_page_config(page_title="Interactive Preprocessor", layout="wide")
st.title("Interactive Preprocessor — produce a clean, ready-to-use CSV")

# -------------------------
# Helper functions
# -------------------------
def load_dataset(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    raise ValueError("Unsupported file type. Upload CSV or Excel.")

def is_id_like(series, uniqueness_ratio=0.95, min_unique=20):
    # If almost all values unique (uniqueness_ratio of rows) AND min unique threshold reached
    nunique = series.nunique(dropna=True)
    if nunique < min_unique:
        return False
    return (nunique / len(series.dropna())) >= uniqueness_ratio

def hash_encode_series(series, num_buckets=1000):
    return series.fillna("__nan__").astype(str).apply(
        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % num_buckets
    )

def freq_encode_series(series):
    vc = series.fillna("__nan__").astype(str).value_counts()
    return series.fillna("__nan__").astype(str).map(vc).astype(float)

def ordinal_encode_series(series):
    # deterministic mapping by sorted unique values
    uniques = sorted(series.dropna().astype(str).unique())
    mapping = {v: i+1 for i, v in enumerate(uniques)}
    return series.fillna("__nan__").astype(str).map(mapping).astype(float)

def iqr_cap_series(series, factor=1.5):
    s = pd.to_numeric(series, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return s.clip(lower, upper)

def safe_get_dummies(df, cols, prefix_sep="_"):
    # Use pandas.get_dummies but check explosion size before applying
    expected_new_cols = sum(df[c].nunique(dropna=True) for c in cols)
    return pd.get_dummies(df, columns=cols, prefix_sep=prefix_sep, dummy_na=False), expected_new_cols

def make_histograms(df, out_dir):
    saved = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(df[c].dropna(), bins=30)
        ax.set_title(f"Histogram: {c}")
        fname = os.path.join(out_dir, f"hist_{c}.png")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
    return saved

def make_boxplots(df, out_dir):
    saved = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.boxplot(df[c].dropna())
        ax.set_title(f"Boxplot: {c}")
        fname = os.path.join(out_dir, f"box_{c}.png")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
    return saved

def make_corr_heatmap(df, out_dir):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0)
    ax.set_title("Correlation heatmap")
    fname = os.path.join(out_dir, "corr_heatmap.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

def make_cat_bars(df, out_dir, top_n=20):
    saved = []
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for c in cat_cols:
        vc = df[c].fillna("__nan__").astype(str).value_counts().iloc[:top_n]
        fig, ax = plt.subplots(figsize=(6, max(3, 0.25*len(vc))))
        ax.barh(vc.index.astype(str), vc.values)
        ax.set_title(f"Counts: {c}")
        fname = os.path.join(out_dir, f"cat_{c}.png")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)
    return saved

# -------------------------
# UI: Upload and inspect
# -------------------------
uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

try:
    df = load_dataset(uploaded)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

st.subheader("Initial preview")
st.write(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")
st.dataframe(df.head())

# make a copy
orig_df = df.copy(deep=True)

# detect columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

st.write("Detected numeric columns:", numeric_cols)
st.write("Detected categorical columns:", categorical_cols)

# basic stats
missing_pct = (df.isna().mean()*100).round(2).sort_values(ascending=False)
st.subheader("Missing values (%) by column")
st.dataframe(missing_pct[missing_pct>0])

# Identify high-cardinality categorical columns
HIGH_CARD_THRESHOLD = st.sidebar.number_input("High-cardinality threshold (unique values)", min_value=20, max_value=100000, value=50, step=5)
LOW_CARD_THRESHOLD = st.sidebar.number_input("Low-cardinality threshold (one-hot cutoff)", min_value=2, max_value=200, value=20, step=1)

high_card_cols = [c for c in categorical_cols if df[c].nunique(dropna=True) > HIGH_CARD_THRESHOLD]
low_card_cols = [c for c in categorical_cols if 0 < df[c].nunique(dropna=True) <= LOW_CARD_THRESHOLD]
medium_card_cols = [c for c in categorical_cols if df[c].nunique(dropna=True) > 0 and (df[c].nunique(dropna=True) > LOW_CARD_THRESHOLD and df[c].nunique(dropna=True) <= HIGH_CARD_THRESHOLD)]

st.write(f"High-cardinality cols (> {HIGH_CARD_THRESHOLD} unique): {high_card_cols}")
st.write(f"Medium-cardinality cols ({LOW_CARD_THRESHOLD+1}..{HIGH_CARD_THRESHOLD}): {medium_card_cols}")
st.write(f"Low-cardinality cols (<= {LOW_CARD_THRESHOLD}): {low_card_cols}")

# -------------------------
# Interactive choices
# -------------------------
st.sidebar.header("Global options")
drop_missing_cols_pct = st.sidebar.slider("Drop columns with missing % >", 0, 100, 40)
apply_iqr = st.sidebar.checkbox("Apply IQR outlier capping to numeric columns", value=True)
iqr_factor = st.sidebar.number_input("IQR factor", min_value=0.1, value=1.5, step=0.1)
scaler_choice = st.sidebar.selectbox("Numeric scaling", ["None", "Standard", "MinMax", "Robust"], index=1)

# Missing value strategy selection (global defaults)
st.sidebar.subheader("Missing values - global default strategies")
numeric_missing_strategy = st.sidebar.selectbox("Numeric missing strategy", ["Median", "Mean", "Constant", "Drop rows"], index=0)
numeric_constant = st.sidebar.number_input("Numeric constant (if Constant)", value=0.0, format="%f")
cat_missing_strategy = st.sidebar.selectbox("Categorical missing strategy", ["Most frequent", "Constant", "Drop rows"], index=0)
cat_constant = st.sidebar.text_input("Categorical constant (if Constant)", value="__MISSING__")

# Per-column override for missing/encoding choices
st.subheader("Per-column decisions (choose per-column options where offered)")
st.markdown("You can override defaults for specific columns below. If you leave per-column widgets untouched the global defaults will apply.")

decision_log = []  # will collect user choices and actions

# per-column missing overrides
col_missing_choices = {}
for c in df.columns:
    miss_pct = (df[c].isna().mean()*100).round(3)
    if miss_pct > 0:
        st.write(f"Missing in `{c}`: {miss_pct}%")
        opt = st.selectbox(f"Missing strategy for `{c}`", ["Use global default", "Median", "Mean", "Constant", "Most frequent", "Drop rows", "Drop column"], key=f"miss_{c}")
        if opt != "Use global default":
            col_missing_choices[c] = opt

# High-cardinality columns handling
high_card_decisions = {}
if high_card_cols:
    st.subheader("High-cardinality columns - choose how to handle each")
    st.markdown("High-cardinality columns can explode memory when one-hot encoded. Choose one of the safer encodings.")
    for c in high_card_cols:
        sample_vals = df[c].astype(str).dropna().unique()[:5].tolist()
        st.write(f"Column `{c}` sample values: {sample_vals}  (unique: {df[c].nunique()})")
        opt = st.selectbox(f"Action for `{c}`", ["Drop column", "Frequency encoding", "Hash encoding (buckets)", "Ordinal encoding"], key=f"hc_{c}")
        high_card_decisions[c] = opt

# Medium-cardinality columns (user can choose to one-hot or fallback)
medium_card_decisions = {}
if medium_card_cols:
    st.subheader("Medium-cardinality columns - choose encoding")
    st.markdown("Medium-card columns can be one-hot encoded if you accept expansion. Otherwise choose ordinal/frequency.")
    for c in medium_card_cols:
        nuniq = df[c].nunique(dropna=True)
        st.write(f"`{c}` unique: {nuniq}")
        opt = st.selectbox(f"Action for `{c}`", ["One-hot (pd.get_dummies)", "Frequency encoding", "Hash encoding", "Ordinal encoding", "Drop column"], key=f"mc_{c}")
        medium_card_decisions[c] = opt

# Low-cardinality columns
low_card_decisions = {}
if low_card_cols:
    st.subheader("Low-cardinality columns - choose encoding")
    for c in low_card_cols:
        st.write(f"`{c}` unique: {df[c].nunique(dropna=True)}")
        opt = st.selectbox(f"Action for `{c}`", ["One-hot (pd.get_dummies)", "Ordinal encoding", "Frequency encoding", "Drop column"], key=f"lc_{c}")
        low_card_decisions[c] = opt

# Numeric scaling per-column override (optional)
scale_override = {}
if numeric_cols:
    st.subheader("Numeric scaling overrides (optional)")
    for c in numeric_cols:
        opt = st.selectbox(f"Scaling for `{c}`", ["Use global", "None", "Standard", "MinMax", "Robust"], key=f"scale_{c}")
        if opt != "Use global":
            scale_override[c] = opt

# Confirm and apply
st.markdown("---")
if st.button("Apply preprocessing and generate outputs"):
    decisions = {
        "timestamp": datetime.now().isoformat(),
        "drop_missing_cols_pct": int(drop_missing_cols_pct),
        "apply_iqr": bool(apply_iqr),
        "iqr_factor": float(iqr_factor),
        "scaler_choice": scaler_choice,
        "numeric_missing_strategy": numeric_missing_strategy,
        "numeric_constant": float(numeric_constant),
        "cat_missing_strategy": cat_missing_strategy,
        "cat_constant": cat_constant,
        "per_column_missing": col_missing_choices,
        "high_cardinality": high_card_decisions,
        "medium_cardinality": medium_card_decisions,
        "low_cardinality": low_card_decisions,
        "scale_overrides": scale_override
    }

    working = df.copy(deep=True)
    report_lines = []
    report_lines.append(f"Original shape: {df.shape}")

    # 1) Drop columns with too many missing values (global)
    miss_pct_all = (working.isna().mean()*100)
    to_drop_missing = miss_pct_all[miss_pct_all > drop_missing_cols_pct].index.tolist()
    if to_drop_missing:
        report_lines.append(f"Dropping columns with >{drop_missing_cols_pct}% missing: {to_drop_missing}")
        working = working.drop(columns=to_drop_missing)

    # 2) Apply per-column missing overrides or global defaults
    # We'll implement: Drop column / Drop rows / constant / median / mean / most frequent
    # First, handle drop rows decisions (collect columns that requested drop rows)
    drop_rows_cols = []
    for c, opt in col_missing_choices.items():
        if opt == "Drop rows":
            drop_rows_cols.append(c)
    if drop_rows_cols:
        # drop any row that has NA in these columns
        before = working.shape
        working = working.dropna(subset=drop_rows_cols)
        report_lines.append(f"Dropped rows with NA in columns: {drop_rows_cols}. New shape: {before} -> {working.shape}")

    # Now handle per-column or global imputation
    for c in list(working.columns):
        # skip if column was dropped
        if c not in working.columns:
            continue
        # per-column override
        opt = col_missing_choices.get(c, "Use global default")
        if opt == "Use global default":
            # choose based on type
            if c in numeric_cols:
                opt = numeric_missing_strategy
            else:
                opt = cat_missing_strategy

        # apply
        if opt == "Drop column":
            if c in working.columns:
                working = working.drop(columns=[c])
                report_lines.append(f"Dropped column {c} per missing-value rule.")
        elif opt == "Drop rows":
            before = working.shape
            working = working.dropna(subset=[c])
            report_lines.append(f"Dropped rows with NA in {c}. New shape {before}->{working.shape}")
        elif opt == "Median":
            if c in working.columns:
                med = working[c].median()
                working[c] = working[c].fillna(med)
                report_lines.append(f"Filled NA in {c} with median = {med}")
        elif opt == "Mean":
            if c in working.columns:
                meanv = working[c].mean()
                working[c] = working[c].fillna(meanv)
                report_lines.append(f"Filled NA in {c} with mean = {meanv}")
        elif opt == "Constant":
            if c in working.columns:
                if c in numeric_cols:
                    working[c] = working[c].fillna(numeric_constant)
                    report_lines.append(f"Filled NA in {c} with numeric constant = {numeric_constant}")
                else:
                    working[c] = working[c].fillna(cat_constant)
                    report_lines.append(f"Filled NA in {c} with categorical constant = {cat_constant}")
        elif opt == "Most frequent":
            if c in working.columns:
                mv = working[c].mode(dropna=True)
                fv = mv.iloc[0] if not mv.empty else ""
                working[c] = working[c].fillna(fv)
                report_lines.append(f"Filled NA in {c} with most frequent = {fv}")
        else:
            # unknown -> skip
            pass

    # 3) High-cardinality treatment
    for c, action in high_card_decisions.items():
        if c not in working.columns:
            continue
        if action == "Drop column" or action == "Drop Column":
            working = working.drop(columns=[c])
            report_lines.append(f"Dropped high-cardinality column {c}")
        elif action == "Frequency encoding" or action == "Frequency Encoding":
            working[c] = freq_encode_series(working[c])
            report_lines.append(f"Frequency encoded {c}")
        elif action == "Hash encoding" or action == "Hash encoding (buckets)":
            # ask bucket count? use 1000 default
            working[c] = hash_encode_series(working[c], num_buckets=1000)
            report_lines.append(f"Hash encoded {c} into 1000 buckets")
        elif action == "Ordinal encoding" or action == "Ordinal encoding":
            working[c] = ordinal_encode_series(working[c])
            report_lines.append(f"Ordinal encoded {c}")
        else:
            report_lines.append(f"No action for {c}")

    # 4) Medium-cardinality
    # If user chose One-hot, check expected expansion; warn and allow fallback
    for c, action in medium_card_decisions.items():
        if c not in working.columns:
            continue
        if action == "One-hot (pd.get_dummies)":
            # estimate expansion
            nuniq = working[c].nunique(dropna=True)
            est_new_cols = nuniq - 1  # one-hot produces N columns roughly
            if est_new_cols > 1000:
                # avoid explosion: convert to frequency instead
                working[c] = freq_encode_series(working[c])
                report_lines.append(f"Column {c} had {nuniq} uniques; too large for one-hot => applied frequency encoding instead.")
            else:
                # safe to one-hot later via get_dummies; mark it so we can apply all low/medium one-hot together
                report_lines.append(f"Marked {c} for one-hot encoding")
        elif action == "Frequency encoding":
            working[c] = freq_encode_series(working[c])
            report_lines.append(f"Frequency encoded {c}")
        elif action == "Hash encoding":
            working[c] = hash_encode_series(working[c], num_buckets=500)
            report_lines.append(f"Hash encoded {c} with 500 buckets")
        elif action == "Ordinal encoding":
            working[c] = ordinal_encode_series(working[c])
            report_lines.append(f"Ordinal encoded {c}")
        elif action == "Drop column":
            working = working.drop(columns=[c])
            report_lines.append(f"Dropped {c} per medium-card decision")

    # 5) Low-cardinality columns
    one_hot_candidates = []
    for c, action in low_card_decisions.items():
        if c not in working.columns:
            continue
        if action == "One-hot (pd.get_dummies)":
            one_hot_candidates.append(c)
        elif action == "Ordinal encoding":
            working[c] = ordinal_encode_series(working[c])
            report_lines.append(f"Ordinal encoded {c}")
        elif action == "Frequency encoding":
            working[c] = freq_encode_series(working[c])
            report_lines.append(f"Frequency encoded {c}")
        elif action == "Drop column":
            working = working.drop(columns=[c])
            report_lines.append(f"Dropped {c} per low-card decision")

    # Apply one-hot to all selected low-card and medium that requested it (we only added low-card to one_hot_candidates)
    if one_hot_candidates:
        # estimate new cols
        est_cols = sum(working[c].nunique(dropna=True) for c in one_hot_candidates)
        if est_cols > 5000:
            # Too many new columns — fall back to frequency encoding and warn
            for c in one_hot_candidates:
                working[c] = freq_encode_series(working[c])
            report_lines.append(f"One-hot candidates would add {est_cols} columns (too many) => applied frequency encoding instead.")
        else:
            # safe to one-hot
            working = pd.get_dummies(working, columns=one_hot_candidates, dummy_na=False)
            report_lines.append(f"One-hot encoded columns: {one_hot_candidates}")

    # 6) Optional IQR capping for numeric columns
    numeric_cols_after = working.select_dtypes(include=[np.number]).columns.tolist()
    if apply_iqr and numeric_cols_after:
        for c in numeric_cols_after:
            before_stats = (working[c].min(), working[c].max())
            working[c] = iqr_cap_series(working[c], factor=iqr_factor)
            after_stats = (working[c].min(), working[c].max())
            report_lines.append(f"IQR capped {c}: min/max {before_stats} -> {after_stats}")

    # 7) Scaling
    # Build scaler object mapping and apply where appropriate
    scaler_map = {}
    # decide global scaler
    def apply_scaler_to_df(df_local, scaler_name, cols):
        if not cols:
            return df_local
        if scaler_name == "None":
            return df_local
        if scaler_name == "Standard":
            scaler = StandardScaler()
        elif scaler_name == "MinMax":
            scaler = MinMaxScaler()
        elif scaler_name == "Robust":
            scaler = RobustScaler()
        else:
            return df_local
        # fit-transform
        df_local[cols] = scaler.fit_transform(df_local[cols].astype(float))
        return df_local

    # Build list of numeric cols to scale
    numeric_cols_now = working.select_dtypes(include=[np.number]).columns.tolist()
    # apply per-column overrides first (collect columns for each method)
    use_global_scaler_cols = []
    per_col_scaling_map = {}
    for c in numeric_cols_now:
        if c in scale_override:
            per_col_scaling_map[c] = scale_override[c]
        else:
            use_global_scaler_cols.append(c)

    # apply per-column scalers
    for c, scl in per_col_scaling_map.items():
        if scl == "None" or scl == "Use global":
            continue
        working = apply_scaler_to_df(working, scl, [c])
        report_lines.append(f"Applied per-column scaler {scl} to {c}")

    # apply global scaler to remaining numeric columns
    if scaler_choice != "None" and use_global_scaler_cols:
        working = apply_scaler_to_df(working, scaler_choice, use_global_scaler_cols)
        report_lines.append(f"Applied global scaler {scaler_choice} to columns: {use_global_scaler_cols}")

    # final cleaned df
    cleaned_df = working.copy()
    report_lines.append(f"Final cleaned shape: {cleaned_df.shape}")

    # Build output folder and plots
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"preproc_run_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # Ensure all subfolders exist
    subfolders = ["orig_hist", "orig_box", "orig_cat", "orig_corr",
                  "clean_hist", "clean_box", "clean_cat", "clean_corr"]
    for sf in subfolders:
        os.makedirs(os.path.join(plots_dir, sf), exist_ok=True)

    # Plots from original (before cleaning) and cleaned
    try:
        orig_hist = make_histograms(orig_df, os.path.join(plots_dir, "orig_hist"))
        orig_box = make_boxplots(orig_df, os.path.join(plots_dir, "orig_box"))
        orig_cat = make_cat_bars(orig_df, os.path.join(plots_dir, "orig_cat"))
        orig_corr = make_corr_heatmap(orig_df, os.path.join(plots_dir, "orig_corr"))
        report_lines.append(f"Generated original dataset plots")
    except Exception as e:
        report_lines.append(f"Failed original plots: {e}")

    try:
        new_hist = make_histograms(cleaned_df, os.path.join(plots_dir, "clean_hist"))
        new_box = make_boxplots(cleaned_df, os.path.join(plots_dir, "clean_box"))
        new_cat = make_cat_bars(cleaned_df, os.path.join(plots_dir, "clean_cat"))
        new_corr = make_corr_heatmap(cleaned_df, os.path.join(plots_dir, "clean_corr"))
        report_lines.append(f"Generated cleaned dataset plots")
    except Exception as e:
        report_lines.append(f"Failed cleaned plots: {e}")

    # Save cleaned csv
    cleaned_csv_path = os.path.join(out_dir, f"cleaned_{ts}.csv")
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    report_lines.append(f"Saved cleaned CSV: {cleaned_csv_path}")

    # Save decisions (pipeline) as a joblib .pkl for later reuse (it stores the decisions dict)
    pipeline_obj = decisions.copy()
    pipeline_obj["report_lines_sample"] = report_lines[:10]
    pipeline_path = os.path.join(out_dir, f"pipeline_decisions_{ts}.pkl")
    dump(pipeline_obj, pipeline_path)
    report_lines.append(f"Saved pipeline decisions (joblib): {pipeline_path}")

    # Save textual report and html
    report_txt_path = os.path.join(out_dir, f"report_{ts}.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    report_lines.append(f"Saved text report: {report_txt_path}")

    # build simple HTML report embedding some plot thumbnails
    html_path = os.path.join(out_dir, f"report_{ts}.html")
    html_lines = ["<html><head><meta charset='utf-8'><title>Preprocessing Report</title></head><body>"]
    html_lines.append(f"<h1>Preprocessing Report</h1><p>Run: {ts}</p>")
    html_lines.append("<h2>Decisions</h2><pre>")
    html_lines.append(json.dumps(decisions, indent=2))
    html_lines.append("</pre>")
    html_lines.append("<h2>Actions / Log</h2><pre>")
    html_lines.append("\n".join(report_lines))
    html_lines.append("</pre>")
    # include some plot thumbnails if exist
    for subfolder in ["orig_hist", "orig_box", "orig_cat", "orig_corr", "clean_hist", "clean_box", "clean_cat", "clean_corr"]:
        folder = os.path.join(plots_dir, subfolder)
        if os.path.isdir(folder):
            for fimg in sorted(os.listdir(folder))[:8]:
                p = os.path.join("plots", subfolder, fimg)
                src = os.path.join(folder, fimg)
                # copy to out_dir/report_plots for relative linking
                rpt_plots_dir = os.path.join(out_dir, "report_plots", subfolder)
                os.makedirs(rpt_plots_dir, exist_ok=True)
                dst = os.path.join(rpt_plots_dir, fimg)
                try:
                    with open(src, "rb") as r, open(dst, "wb") as w:
                        w.write(r.read())
                    html_lines.append(f"<div style='margin:10px'><img src='report_plots/{subfolder}/{fimg}' style='max-width:600px'></div>")
                except Exception:
                    pass
    html_lines.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    report_lines.append(f"Saved HTML report: {html_path}")

    # ZIP everything
    zip_buffer = io.BytesIO()
    zname = f"preproc_results_{ts}.zip"
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # cleaned csv
        zf.write(cleaned_csv_path, arcname=os.path.basename(cleaned_csv_path))
        # pipeline pkl
        zf.write(pipeline_path, arcname=os.path.basename(pipeline_path))
        # txt and html reports
        zf.write(report_txt_path, arcname=os.path.basename(report_txt_path))
        zf.write(html_path, arcname=os.path.basename(html_path))
        # add all report_plots
        rpt_plots_folder = os.path.join(out_dir, "report_plots")
        for root, _, files in os.walk(rpt_plots_folder):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.join("plots", os.path.relpath(full, rpt_plots_folder))
                zf.write(full, arcname=arc)

    zip_buffer.seek(0)

    # show results summary in Streamlit
    st.success("Preprocessing complete")
    st.subheader("Actions taken (sample)")
    st.write("\n".join(report_lines[:200]))

    st.download_button("Download cleaned CSV", data=open(cleaned_csv_path, "rb").read(),
                       file_name=os.path.basename(cleaned_csv_path), mime="text/csv")
    st.download_button("Download pipeline decisions (.pkl)", data=open(pipeline_path, "rb").read(),
                       file_name=os.path.basename(pipeline_path), mime="application/octet-stream")
    st.download_button("Download full results (ZIP)", data=zip_buffer, file_name=zname, mime="application/zip")

    st.info(f"All artifacts are also saved in local folder: {out_dir}")