import base64
import io
import json
import os
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy import stats as scipy_stats

import matplotlib

matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

import ollama

# Seaborn style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#161b22",
    "axes.facecolor": "#1c2129",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "text.color": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#30363d",
    "legend.facecolor": "#1c2129",
    "legend.edgecolor": "#30363d",
})

app = FastAPI(title="Amy - AI File Analysis Agent")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# In-memory store: session_id -> {"files": {file_id: {summary, df, eda, cleaning}}, "active_file": file_id}
current_data: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    session_id: str
    file_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


class ChartRequest(BaseModel):
    session_id: str
    file_id: str
    chart_type: str  # bar, line, pie, scatter, histogram, box, heatmap, area
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    title: Optional[str] = None
    aggregation: str = "sum"  # sum, mean, median, count


class CleaningRequest(BaseModel):
    session_id: str
    file_id: str
    missing_strategy: str = "median"
    outlier_strategy: str = "cap"
    duplicate_strategy: str = "keep"


# ---------------------------------------------------------------------------
# File parsing helpers
# ---------------------------------------------------------------------------


def parse_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def parse_xlsx(file_path: Path) -> pd.DataFrame:
    return pd.read_excel(file_path, engine="openpyxl")


def parse_txt(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    lines = content.splitlines()
    return {
        "filename": file_path.name,
        "type": "text",
        "total_lines": len(lines),
        "total_chars": len(content),
        "preview": "\n".join(lines[:50]),
        "columns": [],
        "row_count": len(lines),
        "stats": {},
    }


def parse_pbix(file_path: Path) -> dict:
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            contents = zf.namelist()
        data_files = [n for n in contents if "DataModel" in n or ".json" in n or "Layout" in n]
        preview_parts = []
        with zipfile.ZipFile(file_path, "r") as zf:
            for name in data_files[:10]:
                try:
                    raw = zf.read(name)
                    try:
                        text = raw.decode("utf-8", errors="replace")
                    except Exception:
                        text = f"[binary, {len(raw)} bytes]"
                    preview_parts.append(f"--- {name} ---\n{text[:2000]}")
                except Exception:
                    preview_parts.append(f"--- {name} ---\n[could not read]")
        return {
            "filename": file_path.name,
            "type": "powerbi",
            "total_entries": len(contents),
            "data_entries": len(data_files),
            "preview": "\n".join(preview_parts[:3]),
            "columns": [],
            "row_count": 0,
            "stats": {"archive_contents": contents[:50]},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse .pbix file: {str(e)}")


def _build_dataframe_summary(df: pd.DataFrame, file_path: Path) -> dict:
    row_count, col_count = df.shape
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    col_stats = {}
    for col in columns:
        dtype = str(df[col].dtype)
        nulls = int(df[col].isna().sum())
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats[col] = {
                "dtype": dtype,
                "nulls": nulls,
                "null_pct": round(nulls / row_count * 100, 1) if row_count else 0,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "mean": round(float(df[col].mean()), 2) if not df[col].isna().all() else None,
                "median": round(float(df[col].median()), 2) if not df[col].isna().all() else None,
                "std": round(float(df[col].std()), 2) if not df[col].isna().all() else None,
            }
        else:
            unique = int(df[col].nunique())
            col_stats[col] = {
                "dtype": dtype,
                "nulls": nulls,
                "null_pct": round(nulls / row_count * 100, 1) if row_count else 0,
                "unique_values": unique,
            }

    preview_df = df.head(20).fillna("").astype(str)
    preview_rows = preview_df.to_dict(orient="records")

    return {
        "filename": file_path.name,
        "type": "tabular",
        "row_count": row_count,
        "col_count": col_count,
        "columns": columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "stats": col_stats,
        "preview_rows": preview_rows,
        "preview": "",
    }


# ---------------------------------------------------------------------------
# EDA (Exploratory Data Analysis)
# ---------------------------------------------------------------------------


def compute_eda(df: pd.DataFrame) -> dict:
    """Run comprehensive EDA and return structured results."""
    row_count, col_count = df.shape
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    # --- Data Quality ---
    missing = df.isna().sum().to_dict()
    missing_pct = (df.isna().sum() / row_count * 100).round(1).to_dict()
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    missing_by_row = df.isna().sum(axis=1)
    rows_with_missing = int((missing_by_row > 0).sum())
    rows_complete = int((missing_by_row == 0).sum())

    # --- Outlier detection (IQR) ---
    outliers = {}
    outlier_rows = set()
    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (numeric_df[col] < lower) | (numeric_df[col] > upper)
        outlier_count = int(mask.sum())
        outlier_rows.update(df.index[mask].tolist())
        outliers[col] = {
            "count": outlier_count,
            "pct": round(outlier_count / row_count * 100, 1) if row_count else 0,
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
            "q1": round(float(q1), 2) if not pd.isna(q1) else None,
            "q3": round(float(q3), 2) if not pd.isna(q3) else None,
            "iqr": round(float(iqr), 2) if not pd.isna(iqr) else None,
        }

    # --- Distribution metrics ---
    distributions = {}
    for col in numeric_df.columns:
        clean = numeric_df[col].dropna()
        if len(clean) > 2:
            skew = round(float(scipy_stats.skew(clean)), 3)
            kurt = round(float(scipy_stats.kurtosis(clean)), 3)
            q10 = round(float(clean.quantile(0.10)), 2)
            q25 = round(float(clean.quantile(0.25)), 2)
            q50 = round(float(clean.quantile(0.50)), 2)
            q75 = round(float(clean.quantile(0.75)), 2)
            q90 = round(float(clean.quantile(0.90)), 2)
            distributions[col] = {"skewness": skew, "kurtosis": kurt}
            distributions[col].update({
                "q10": q10,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q90": q90,
            })

    # --- Categorical insights ---
    category_insights = {}
    for col in categorical_df.columns[:20]:
        clean = categorical_df[col].dropna().astype(str)
        top_values = clean.value_counts().head(5)
        category_insights[col] = {
            "unique_values": int(categorical_df[col].nunique(dropna=True)),
            "top_values": [
                {"value": str(idx), "count": int(val), "pct": round(val / row_count * 100, 1) if row_count else 0}
                for idx, val in top_values.items()
            ],
        }

    # --- Correlation matrix (top pairs) ---
    correlations = []
    strong_correlations = []
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = round(float(corr_matrix.iloc[i, j]), 3)
                correlations.append({
                    "col1": corr_matrix.columns[i],
                    "col2": corr_matrix.columns[j],
                    "correlation": val,
                })
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        correlations = correlations[:15]
        strong_correlations = [c for c in correlations if abs(c["correlation"]) >= 0.7][:10]

    # --- Missingness by column ---
    missing_rank = sorted(
        [{"column": k, "count": int(v), "pct": missing_pct[k]} for k, v in missing.items()],
        key=lambda x: (x["pct"], x["count"]),
        reverse=True,
    )

    # --- Shape and completeness ---
    completeness_pct = round(rows_complete / row_count * 100, 1) if row_count else 0
    missing_rows_pct = round(rows_with_missing / row_count * 100, 1) if row_count else 0

    # --- Simple data types summary ---
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()

    # --- Data quality score ---
    quality_deductions = 0
    quality_deductions += min(total_missing / max(row_count * col_count, 1) * 40, 40)
    quality_deductions += min(duplicate_rows / max(row_count, 1) * 25, 25)
    quality_deductions += len(constant_cols) * 5
    for o in outliers.values():
        quality_deductions += min(o["pct"] / 100 * 10, 10)
    quality_score = round(max(0, 100 - quality_deductions), 1)
    quality_flags = []
    if missing_pct and max(missing_pct.values()) > 30:
        quality_flags.append("Some columns have very high missingness")
    if duplicate_rows > 0:
        quality_flags.append("Duplicate rows detected")
    if strong_correlations:
        quality_flags.append("Strong correlations found between some numeric features")
    if constant_cols:
        quality_flags.append("Constant columns present")

    # --- Describe stats ---
    describe = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().round(2)
        describe = desc.to_dict()

    percentile_summary = {}
    if not numeric_df.empty:
        for col in numeric_df.columns:
            clean = numeric_df[col].dropna()
            if not clean.empty:
                percentile_summary[col] = {
                    "min": round(float(clean.min()), 2),
                    "p10": round(float(clean.quantile(0.10)), 2),
                    "p25": round(float(clean.quantile(0.25)), 2),
                    "median": round(float(clean.quantile(0.50)), 2),
                    "p75": round(float(clean.quantile(0.75)), 2),
                    "p90": round(float(clean.quantile(0.90)), 2),
                    "max": round(float(clean.max()), 2),
                }

    return {
        "row_count": row_count,
        "col_count": col_count,
        "numeric_col_count": numeric_df.shape[1],
        "categorical_col_count": col_count - numeric_df.shape[1],
        "quality_score": quality_score,
        "quality_flags": quality_flags,
        "total_missing": total_missing,
        "missing_pct": round(total_missing / max(row_count * col_count, 1) * 100, 1),
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_pct": missing_rows_pct,
        "rows_complete": rows_complete,
        "rows_complete_pct": completeness_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / max(row_count, 1) * 100, 1),
        "constant_columns": constant_cols,
        "missing_by_column": {k: {"count": int(v), "pct": missing_pct[k]} for k, v in missing.items()},
        "missing_rank": missing_rank,
        "outliers": outliers,
        "distributions": distributions,
        "percentile_summary": percentile_summary,
        "categorical_insights": category_insights,
        "correlations": correlations,
        "strong_correlations": strong_correlations,
        "dtype_counts": dtype_counts,
        "outlier_rows": len(outlier_rows),
        "describe": describe,
    }


def compute_cleaning_issues(df: pd.DataFrame) -> dict:
    """Identify data quality issues and prepare a cleaning report."""
    row_count = len(df)
    issues = []

    # Missing values
    missing = df.isna().sum()
    for col in df.columns:
        if missing[col] > 0:
            pct = round(missing[col] / row_count * 100, 1)
            severity = "high" if pct > 20 else ("medium" if pct > 5 else "low")
            issues.append({
                "type": "missing_values",
                "column": col,
                "detail": f"{int(missing[col])} missing values ({pct}%)",
                "severity": severity,
            })

    # Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append({
            "type": "duplicates",
            "column": "—",
            "detail": f"{dup_count} duplicate rows ({round(dup_count / row_count * 100, 1)}%)",
            "severity": "high" if dup_count / row_count > 0.1 else "medium",
        })

    # Outliers (numeric columns)
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        if outlier_count > 0:
            pct = round(outlier_count / row_count * 100, 1)
            issues.append({
                "type": "outliers",
                "column": col,
                "detail": f"{outlier_count} outliers ({pct}%), bounds: [{round(lower, 2)}, {round(upper, 2)}]",
                "severity": "high" if pct > 15 else ("medium" if pct > 5 else "low"),
            })

    # Constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues.append({
                "type": "constant_column",
                "column": col,
                "detail": "Single unique value — no analytical value",
                "severity": "low",
            })

    # Mixed types (check for inconsistent types across rows — simplified)
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(50)
        numeric_like = sum(1 for v in sample if isinstance(v, str) and v.replace(".", "", 1).replace("-", "", 1).isdigit())
        if 0 < numeric_like < len(sample):
            issues.append({
                "type": "mixed_types",
                "column": col,
                "detail": "Column appears to contain mixed numeric/text values",
                "severity": "medium",
            })

    return {
        "total_issues": len(issues),
        "issues": issues,
        "issues_by_severity": {
            "high": len([i for i in issues if i["severity"] == "high"]),
            "medium": len([i for i in issues if i["severity"] == "medium"]),
            "low": len([i for i in issues if i["severity"] == "low"]),
        },
    }


def _clean_numeric_series(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "median":
        fill_value = series.median()
        if pd.isna(fill_value):
            fill_value = 0
        return series.fillna(fill_value)
    if strategy == "mean":
        fill_value = series.mean()
        if pd.isna(fill_value):
            fill_value = 0
        return series.fillna(fill_value)
    if strategy == "interpolate":
        return series.interpolate(limit_direction="both")
    if strategy == "drop":
        return series
    return series.fillna(series.median() if not pd.isna(series.median()) else 0)


def _clean_categorical_series(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "mode":
        mode_vals = series.mode(dropna=True)
        fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
        return series.fillna(fill_value)
    if strategy == "ffill":
        return series.fillna(method="ffill").fillna(method="bfill")
    if strategy == "bfill":
        return series.fillna(method="bfill").fillna(method="ffill")
    if strategy == "drop":
        return series
    mode_vals = series.mode(dropna=True)
    fill_value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
    return series.fillna(fill_value)


def clean_tabular_data(df: pd.DataFrame, missing_strategy: str, outlier_strategy: str, duplicate_strategy: str) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    report = {
        "missing_strategy": missing_strategy,
        "outlier_strategy": outlier_strategy,
        "duplicate_strategy": duplicate_strategy,
        "steps": [],
    }

    missing_before = int(cleaned.isna().sum().sum())
    if missing_strategy == "drop":
        cleaned = cleaned.dropna()
        report["steps"].append({"type": "missing_values", "action": "dropped rows with missing values"})
    elif missing_strategy == "interpolate":
        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            cleaned[col] = _clean_numeric_series(cleaned[col], "interpolate")
        for col in cleaned.columns.difference(num_cols):
            cleaned[col] = _clean_categorical_series(cleaned[col], "mode")
        report["steps"].append({"type": "missing_values", "action": "interpolated numeric columns and filled categorical columns with mode"})
    else:
        for col in cleaned.columns:
            if cleaned[col].isna().any():
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = _clean_numeric_series(cleaned[col], missing_strategy)
                else:
                    cleaned[col] = _clean_categorical_series(cleaned[col], missing_strategy if missing_strategy in {"mode", "ffill", "bfill"} else "mode")
        report["steps"].append({"type": "missing_values", "action": f"filled missing values using {missing_strategy}"})
    missing_after = int(cleaned.isna().sum().sum())

    duplicate_before = int(cleaned.duplicated().sum())
    if duplicate_strategy == "drop":
        cleaned = cleaned.drop_duplicates()
        report["steps"].append({"type": "duplicates", "action": "dropped duplicate rows"})
    else:
        report["steps"].append({"type": "duplicates", "action": "kept duplicate rows"})
    duplicate_after = int(cleaned.duplicated().sum())

    outlier_counts = {}
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = cleaned[col]
        clean_series = series.dropna()
        if clean_series.empty:
            continue
        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (cleaned[col] < lower) | (cleaned[col] > upper)
        count = int(mask.sum())
        if count <= 0:
            continue
        outlier_counts[col] = count
        if outlier_strategy == "remove":
            cleaned = cleaned.loc[~mask].copy()
        elif outlier_strategy == "cap":
            cleaned.loc[cleaned[col] < lower, col] = lower
            cleaned.loc[cleaned[col] > upper, col] = upper
        elif outlier_strategy == "median":
            cleaned.loc[mask, col] = clean_series.median()
    if outlier_counts:
        action = {
            "remove": "removed rows containing numeric outliers",
            "cap": "capped numeric outliers to IQR bounds",
            "median": "replaced numeric outliers with the column median",
        }.get(outlier_strategy, "handled numeric outliers")
        report["steps"].append({"type": "outliers", "action": action, "columns": outlier_counts})
    else:
        report["steps"].append({"type": "outliers", "action": "no numeric outliers found"})

    cleaned = cleaned.reset_index(drop=True)
    report["summary"] = {
        "missing_before": missing_before,
        "missing_after": missing_after,
        "duplicate_before": duplicate_before,
        "duplicate_after": duplicate_after,
        "outliers_handled": outlier_counts,
        "rows_before": len(df),
        "rows_after": len(cleaned),
        "columns": len(cleaned.columns),
    }
    return cleaned, report


# ---------------------------------------------------------------------------
# Chart generation — IMPROVED
# ---------------------------------------------------------------------------


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _prepare_chart_data(df: pd.DataFrame, x_col: str, y_col: Optional[str], aggregation: str, limit: int = 20) -> pd.Series:
    """Aggregate chart data safely. Returns a Series with index as x labels and values as y."""
    # Drop rows where x_col is NaN
    data = df.dropna(subset=[x_col]).copy()
    if data.empty:
        raise HTTPException(status_code=400, detail="No valid rows available for the selected chart columns")

    if y_col:
        # Ensure y_col is numeric
        if not pd.api.types.is_numeric_dtype(data[y_col]):
            data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
        data = data.dropna(subset=[y_col])
        if data.empty:
            raise HTTPException(status_code=400, detail=f"Column '{y_col}' must contain numeric values")

        # Group by x_col and aggregate y_col
        if aggregation == "mean":
            series = data.groupby(x_col)[y_col].mean()
        elif aggregation == "median":
            series = data.groupby(x_col)[y_col].median()
        elif aggregation == "count":
            series = data.groupby(x_col)[y_col].count()
        else:  # sum
            series = data.groupby(x_col)[y_col].sum()

        series = series.sort_values(ascending=False).head(limit)
    else:
        # No y_col: count occurrences of each x value
        series = data[x_col].value_counts().head(limit)

    return series


def _format_axis_labels(ax, labels, fontsize=8):
    """Apply common axis label formatting."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(label)[:35] for label in labels], rotation=35, ha="right", fontsize=fontsize)


def _add_bar_labels(ax, rects, fontsize=7):
    """Add value labels on top of bars."""
    for rect in rects:
        height = rect.get_height()
        if pd.isna(height) or height == 0:
            continue
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=fontsize, color="#8b949e")


def generate_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: Optional[str], title: Optional[str], aggregation: str = "sum") -> str:
    """Generate a chart and return as base64 PNG."""
    aggregation = aggregation if aggregation in {"sum", "mean", "median", "count"} else "sum"
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # ─── BAR CHART ────────────────────────────────────────────────────
    if chart_type == "bar":
        agg = _prepare_chart_data(df, x_col, y_col, aggregation)
        colors = [sns.color_palette("Blues_r", len(agg))[i] for i in range(len(agg))]
        bars = ax.bar(range(len(agg)), agg.values, color="#58a6ff", edgecolor="#30363d", linewidth=0.5, alpha=0.9)
        _format_axis_labels(ax, agg.index)
        ax.set_ylabel(f"{aggregation.title()} of {y_col}" if y_col else "Count")
        ax.set_xlabel(x_col)
        _add_bar_labels(ax, bars)

    # ─── LINE CHART ───────────────────────────────────────────────────
    elif chart_type == "line":
        if not y_col:
            raise HTTPException(status_code=400, detail="Line chart requires y_column")
        agg = _prepare_chart_data(df, x_col, y_col, aggregation, limit=60)
        # Sort by index for proper line progression
        agg = agg.sort_index()
        xs = range(len(agg))
        ax.plot(xs, agg.values, color="#3fb950", linewidth=2, marker="o", markersize=4, alpha=0.85)
        ax.fill_between(xs, agg.values, alpha=0.08, color="#3fb950")
        _format_axis_labels(ax, agg.index)
        ax.set_ylabel(f"{aggregation.title()} of {y_col}")
        ax.set_xlabel(x_col)
        # Add value labels on points (sparse)
        step = max(1, len(agg) // 10)
        for i in range(0, len(agg), step):
            ax.annotate(f'{agg.values[i]:.1f}', (xs[i], agg.values[i]),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=6, color="#8b949e")

    # ─── AREA CHART ───────────────────────────────────────────────────
    elif chart_type == "area":
        if not y_col:
            raise HTTPException(status_code=400, detail="Area chart requires y_column")
        agg = _prepare_chart_data(df, x_col, y_col, aggregation, limit=60)
        agg = agg.sort_index()
        xs = range(len(agg))
        ax.fill_between(xs, agg.values, alpha=0.3, color="#58a6ff")
        ax.plot(xs, agg.values, color="#58a6ff", linewidth=2, alpha=0.9)
        _format_axis_labels(ax, agg.index)
        ax.set_ylabel(f"{aggregation.title()} of {y_col}")
        ax.set_xlabel(x_col)

    # ─── PIE CHART ────────────────────────────────────────────────────
    elif chart_type == "pie":
        if y_col:
            # If y_col is provided, aggregate y by x for pie slices
            agg = _prepare_chart_data(df, x_col, y_col, aggregation, limit=10)
        else:
            # Just count x values
            agg = df[x_col].value_counts().head(10)
        if agg.empty or agg.sum() == 0:
            raise HTTPException(status_code=400, detail="No data available for pie chart")
        colors = sns.color_palette("muted", len(agg))
        wedges, texts, autotexts = ax.pie(
            agg.values, labels=agg.index, autopct="%1.1f%%",
            colors=colors, textprops={"fontsize": 7},
            pctdistance=0.75, labeldistance=1.05,
        )
        for at in autotexts:
            at.set_color("#e6edf3")
        for t in texts:
            t.set_fontsize(7)

    # ─── SCATTER CHART ────────────────────────────────────────────────
    elif chart_type == "scatter":
        if not y_col:
            raise HTTPException(status_code=400, detail="Scatter chart requires y_column")
        scatter_df = df[[x_col, y_col]].dropna()
        if scatter_df.empty:
            raise HTTPException(status_code=400, detail="No valid data for scatter plot after removing missing values")
        # Convert to numeric if needed
        scatter_df[x_col] = pd.to_numeric(scatter_df[x_col], errors="coerce")
        scatter_df[y_col] = pd.to_numeric(scatter_df[y_col], errors="coerce")
        scatter_df = scatter_df.dropna()
        if scatter_df.empty:
            raise HTTPException(status_code=400, detail="Columns must contain numeric values for scatter plot")
        # Limit points for performance
        if len(scatter_df) > 5000:
            scatter_df = scatter_df.sample(5000, random_state=42)
        ax.scatter(scatter_df[x_col], scatter_df[y_col], alpha=0.5, c="#58a6ff", edgecolors="none", s=25)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.tick_params(axis="x", rotation=35)

    # ─── HISTOGRAM ────────────────────────────────────────────────────
    elif chart_type == "histogram":
        hist_data = df[x_col].dropna()
        if hist_data.empty:
            raise HTTPException(status_code=400, detail=f"No valid data in column '{x_col}' for histogram")
        hist_data = pd.to_numeric(hist_data, errors="coerce").dropna()
        if hist_data.empty:
            raise HTTPException(status_code=400, detail=f"Column '{x_col}' must contain numeric values for histogram")
        # Use Freedman-Diaconis rule for bins
        q25, q75 = hist_data.quantile(0.25), hist_data.quantile(0.75)
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(hist_data) ** (1/3)) if iqr > 0 else 1
        bins = max(10, min(80, int((hist_data.max() - hist_data.min()) / max(bin_width, 1))))
        n, bins_edges, patches = ax.hist(hist_data, bins=bins, color="#d2991d", edgecolor="#30363d", alpha=0.8)
        ax.set_ylabel("Frequency")
        ax.set_xlabel(x_col)

    # ─── BOX PLOT ────────────────────────────────────────────────────
    elif chart_type == "box":
        if y_col and x_col:
            # Box plot of y_col grouped by x_col categories
            num_cats = 12  # limit categories
            top_cats = df[x_col].value_counts().head(num_cats).index.tolist()
            plot_data = []
            valid_labels = []
            for cat in top_cats:
                vals = df[df[x_col] == cat][y_col].dropna()
                vals = pd.to_numeric(vals, errors="coerce").dropna()
                if len(vals) > 0:
                    plot_data.append(vals)
                    valid_labels.append(str(cat)[:25])
            if not plot_data:
                raise HTTPException(status_code=400, detail="No valid data for box plot")
            bp = ax.boxplot(plot_data, labels=valid_labels, patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='#3fb950', markersize=4))
            for patch in bp["boxes"]:
                patch.set_facecolor("#58a6ff")
                patch.set_alpha(0.7)
            for median_line in bp["medians"]:
                median_line.set_color("#d2991d")
                median_line.set_linewidth(2)
            ax.set_ylabel(y_col)
            ax.set_xlabel(x_col)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
        else:
            # Single column box plot
            vals = pd.to_numeric(df[x_col].dropna(), errors="coerce").dropna()
            if vals.empty:
                raise HTTPException(status_code=400, detail=f"No valid numeric data in '{x_col}' for box plot")
            bp = ax.boxplot(vals, labels=[x_col], patch_artist=True, showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='#3fb950', markersize=5))
            for patch in bp["boxes"]:
                patch.set_facecolor("#58a6ff")
                patch.set_alpha(0.7)
            for median_line in bp["medians"]:
                median_line.set_color("#d2991d")
                median_line.set_linewidth(2)

    # ─── HEATMAP ──────────────────────────────────────────────────────
    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if numeric_df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for heatmap")
        if numeric_df.shape[1] > 20:
            # Pick top 20 columns with highest variance
            variances = numeric_df.var()
            top_cols = variances.nlargest(20).index
            numeric_df = numeric_df[top_cols]
        corr = numeric_df.corr()
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        # Add correlation values as annotations
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.5 else "#8b949e"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)

    else:
        raise HTTPException(status_code=400, detail=f"Unknown chart type: {chart_type}")

    # ─── Title ──────────────────────────────────────────────────────
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    elif chart_type != "heatmap":
        default_title = f"{chart_type.title()} of {x_col}"
        if y_col:
            default_title += f" by {aggregation}({y_col})"
        ax.set_title(default_title, fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    return _fig_to_base64(fig)


def _get_file_entry(session_id: str, file_id: str) -> dict:
    """Look up a file entry within a session."""
    session = current_data.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    entry = session.get("files", {}).get(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail="File not found in session.")
    return entry


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), session_id: Optional[str] = Form(None)):
    """Upload a file. Creates a new session or adds to existing one."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()
    allowed = {".csv", ".txt", ".xlsx", ".pbix"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed)}")

    safe_name = Path(file.filename).name
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB.")

    # Use existing session or create new one
    if session_id and session_id in current_data:
        sid = session_id
    else:
        sid = uuid.uuid4().hex[:8]

    file_id = uuid.uuid4().hex[:6]
    stored_name = f"{sid}_{file_id}_{safe_name}"
    file_path = UPLOAD_DIR / stored_name
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        if ext == ".csv":
            df = parse_csv(file_path)
            summary = _build_dataframe_summary(df, file_path)
            eda = compute_eda(df)
            cleaning = compute_cleaning_issues(df)
            entry = {"summary": summary, "df": df, "eda": eda, "cleaning": cleaning}
            result = {**summary, "eda": eda, "cleaning": cleaning, "session_id": sid, "file_id": file_id}

        elif ext == ".xlsx":
            df = parse_xlsx(file_path)
            summary = _build_dataframe_summary(df, file_path)
            eda = compute_eda(df)
            cleaning = compute_cleaning_issues(df)
            entry = {"summary": summary, "df": df, "eda": eda, "cleaning": cleaning}
            result = {**summary, "eda": eda, "cleaning": cleaning, "session_id": sid, "file_id": file_id}

        elif ext == ".txt":
            result = parse_txt(file_path)
            entry = {"summary": result, "df": None, "eda": None, "cleaning": None}
            result["session_id"] = sid
            result["file_id"] = file_id

        elif ext == ".pbix":
            result = parse_pbix(file_path)
            entry = {"summary": result, "df": None, "eda": None, "cleaning": None}
            result["session_id"] = sid
            result["file_id"] = file_id
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")

    # Store in session
    if sid not in current_data:
        current_data[sid] = {"files": {}, "active_file": file_id}
    current_data[sid]["files"][file_id] = entry
    current_data[sid]["active_file"] = file_id

    # Include file count
    result["file_count"] = len(current_data[sid]["files"])

    return JSONResponse(content=result)


@app.post("/api/summary")
async def get_summary(session_id: str = Query(...), file_id: str = Query(...)):
    """Return full summary (preview_rows, columns, stats) for a file."""
    entry = _get_file_entry(session_id, file_id)
    summary = entry["summary"]
    eda = entry.get("eda") or (compute_eda(entry["df"]) if entry["df"] is not None else None)
    cleaning = entry.get("cleaning") or (compute_cleaning_issues(entry["df"]) if entry["df"] is not None else None)
    return JSONResponse(content={**summary, "eda": eda, "cleaning": cleaning, "session_id": session_id, "file_id": file_id})


@app.post("/api/session/{session_id}")
async def get_session(session_id: str):
    """Return all files in a session."""
    session = current_data.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    files = []
    for fid, entry in session.get("files", {}).items():
        s = entry["summary"]
        files.append({
            "file_id": fid,
            "filename": s.get("filename", "?"),
            "type": s.get("type", "?"),
            "row_count": s.get("row_count", 0),
            "col_count": s.get("col_count", 0),
        })
    return JSONResponse(content={"session_id": session_id, "active_file": session.get("active_file"), "files": files})


@app.post("/api/eda")
async def get_eda(session_id: str = Query(...), file_id: str = Query(...)):
    """Return full EDA results for a file in a session."""
    entry = _get_file_entry(session_id, file_id)
    if entry["df"] is None:
        raise HTTPException(status_code=404, detail="No tabular data found. Upload a CSV/XLSX first.")
    eda = compute_eda(entry["df"])
    return JSONResponse(content=eda)


@app.post("/api/cleaning")
async def get_cleaning_suggestions(session_id: str = Query(...), file_id: str = Query(...)):
    """Return data quality issues and AI-generated cleaning recommendations."""
    entry = _get_file_entry(session_id, file_id)
    if entry["df"] is None:
        raise HTTPException(status_code=404, detail="No tabular data found. Upload first.")
    df = entry["df"]
    cleaning = compute_cleaning_issues(df)
    summary = entry["summary"]

    # Get AI cleaning suggestions
    context = _build_eda_context(summary, cleaning)
    prompt = (
        f"Here is a data quality report for '{summary['filename']}':\n\n{context}\n\n"
        f"Based on these issues, provide concise, actionable cleaning recommendations. "
        f"For each issue type (missing values, duplicates, outliers, etc.), suggest:\n"
        f"1. A specific strategy to fix it\n"
        f"2. Any caveats or things to watch out for\n\n"
        f"Format as a clear, bulleted list. Be practical and specific."
    )

    ai_suggestions = ""
    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {"role": "system", "content": "You are a data cleaning expert. Give practical, specific advice."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_predict": 800},
        )
        ai_suggestions = response["message"]["content"]
    except Exception as e:
        ai_suggestions = f"(Could not generate AI suggestions: {e})"

    return JSONResponse(content={**cleaning, "ai_suggestions": ai_suggestions})


@app.post("/api/cleaning/apply")
async def apply_cleaning(req: CleaningRequest):
    entry = _get_file_entry(req.session_id, req.file_id)
    if entry["df"] is None:
        raise HTTPException(status_code=404, detail="No tabular data found. Upload first.")

    missing_strategy = req.missing_strategy if req.missing_strategy in {"median", "mean", "mode", "interpolate", "drop", "ffill", "bfill"} else "median"
    outlier_strategy = req.outlier_strategy if req.outlier_strategy in {"cap", "remove", "median", "keep"} else "cap"
    duplicate_strategy = req.duplicate_strategy if req.duplicate_strategy in {"keep", "drop"} else "keep"

    cleaned_df, report = clean_tabular_data(entry["df"], missing_strategy, outlier_strategy, duplicate_strategy)
    entry["df"] = cleaned_df
    entry["summary"] = _build_dataframe_summary(cleaned_df, UPLOAD_DIR / entry["summary"]["filename"])
    entry["eda"] = compute_eda(cleaned_df)
    entry["cleaning"] = compute_cleaning_issues(cleaned_df)
    session = current_data.get(req.session_id)
    if session:
        session["active_file"] = req.file_id

    return JSONResponse(content={
        "message": "Cleaning applied successfully.",
        "report": report,
        "summary": entry["summary"],
        "eda": entry["eda"],
        "cleaning": entry["cleaning"],
    })


@app.post("/api/chart")
async def get_chart(req: ChartRequest):
    """Generate a chart and return it as base64 PNG."""
    entry = _get_file_entry(req.session_id, req.file_id)
    if entry["df"] is None:
        raise HTTPException(status_code=404, detail="No tabular data found. Upload first.")
    df = entry["df"]

    x_col = req.x_column
    y_col = req.y_column

    if not x_col:
        raise HTTPException(status_code=400, detail="x_column is required")

    if x_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{x_col}' not found")
    if y_col and y_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{y_col}' not found")

    b64 = generate_chart(df, req.chart_type, x_col, y_col, req.title, req.aggregation)
    return JSONResponse(content={"image": f"data:image/png;base64,{b64}"})


@app.get("/api/download")
async def download_cleaned(
    session_id: str = Query(...),
    file_id: str = Query(...),
    format: str = Query("csv"),
):
    """Download the cleaned DataFrame as CSV or XLSX."""
    entry = _get_file_entry(session_id, file_id)
    if entry["df"] is None:
        raise HTTPException(status_code=404, detail="No tabular data found. Upload & clean first.")

    df: pd.DataFrame = entry["df"]
    filename_base = Path(entry["summary"]["filename"]).stem

    if format == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned")
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}_cleaned.xlsx"'},
        )

    # Default: CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename_base}_cleaned.csv"'},
    )


@app.post("/api/chart-suggestions")
async def suggest_charts(session_id: str = Query(...), file_id: str = Query(...)):
    """Use AI to suggest which charts would be most insightful for the data."""
    entry = _get_file_entry(session_id, file_id)
    if entry["df"] is None:
        raise HTTPException(status_code=404, detail="No tabular data found. Upload first.")

    summary = entry["summary"]
    context = _build_context_for_llm(summary)

    prompt = (
        f"Here is a summary of a dataset named '{summary['filename']}':\n\n{context}\n\n"
        f"Suggest 4-6 specific charts/visualizations that would be most insightful for this data. "
        f"For each suggestion, specify:\n"
        f"- chart_type: one of [bar, line, pie, scatter, histogram, box, heatmap, area]\n"
        f"- x_column: the column name for x-axis\n"
        f"- y_column: the column name for y-axis (null if not applicable)\n"
        f"- title: a descriptive title\n"
        f"- reason: why this chart would be valuable\n\n"
        f"Return your answer as a JSON array of objects. Only return valid JSON, nothing else."
    )

    suggestions = []
    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Return only valid JSON arrays."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.4, "num_predict": 1200},
        )
        raw = response["message"]["content"]
        # Try to extract JSON from the response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:].strip()
            if raw.endswith("```"):
                raw = raw[:-3]
        suggestions = json.loads(raw)
    except Exception as e:
        # Fallback: return basic suggestions based on data types
        suggestions = _fallback_chart_suggestions(summary)

    return JSONResponse(content={"suggestions": suggestions})


def _fallback_chart_suggestions(summary: dict) -> list:
    """Generate basic chart suggestions without AI."""
    suggestions = []
    num_cols = summary.get("numeric_columns", [])
    cat_cols = summary.get("categorical_columns", [])

    if num_cols and cat_cols:
        suggestions.append({
            "chart_type": "bar", "x_column": cat_cols[0], "y_column": num_cols[0],
            "title": f"Average {num_cols[0]} by {cat_cols[0]}",
            "reason": "Shows how a numeric metric varies across categories."
        })
    if len(num_cols) >= 1:
        suggestions.append({
            "chart_type": "histogram", "x_column": num_cols[0], "y_column": None,
            "title": f"Distribution of {num_cols[0]}",
            "reason": "Reveals the shape and spread of the data."
        })
    if len(cat_cols) >= 1:
        suggestions.append({
            "chart_type": "pie", "x_column": cat_cols[0], "y_column": None,
            "title": f"Proportion of {cat_cols[0]}",
            "reason": "Shows the relative composition of categories."
        })
    if len(num_cols) >= 2:
        suggestions.append({
            "chart_type": "scatter", "x_column": num_cols[0], "y_column": num_cols[1],
            "title": f"{num_cols[1]} vs {num_cols[0]}",
            "reason": "Reveals relationship between two numeric variables."
        })
    if len(num_cols) >= 2:
        suggestions.append({
            "chart_type": "heatmap", "x_column": num_cols[0], "y_column": num_cols[1],
            "title": "Correlation Heatmap",
            "reason": "Shows how all numeric variables relate to each other."
        })
    if len(num_cols) >= 1 and len(cat_cols) >= 1:
        suggestions.append({
            "chart_type": "box", "x_column": cat_cols[0], "y_column": num_cols[0],
            "title": f"Distribution of {num_cols[0]} by {cat_cols[0]}",
            "reason": "Compares distributions across categories."
        })
    return suggestions


@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = current_data.get(req.session_id)
    if not session or not session.get("files"):
        raise HTTPException(status_code=404, detail="No data found. Please upload first.")

    # Build context: include active file details + summary of all files
    files = session["files"]
    active_fid = req.file_id or session.get("active_file")
    active_entry = files.get(active_fid) if active_fid else None

    context_lines = [f"Session has {len(files)} file(s):"]
    for fid, entry in files.items():
        s = entry["summary"]
        marker = " ← ACTIVE" if fid == active_fid else ""
        context_lines.append(f"  [{fid}] {s.get('filename','?')} ({s.get('type','?')}, {s.get('row_count','?')} rows){marker}")

    if active_entry:
        summary = active_entry["summary"]
        context_lines.append(f"\n--- Active file details: {summary['filename']} ---")
        if summary.get("type") == "tabular" and active_entry["df"] is not None:
            context_lines.append(_build_context_for_llm(summary))
        else:
            context_lines.append(_build_context_for_llm(summary))

    context = "\n".join(context_lines)

    system_prompt = (
        "You are Amy, an expert data analyst assistant. "
        "You help users understand and analyze their data files. "
        "Answer questions based on the provided data context. "
        "Be concise, insightful, and helpful. "
        "If the user asks something not covered by the data, say so politely. "
        "Suggest relevant analyses and visualizations when appropriate."
    )

    user_prompt = (
        f"Here is the data context:\n\n{context}\n\n"
        f"User question: {req.message}\n\n"
        f"Please answer based on the data above."
    )

    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.3, "num_predict": 1024},
        )
        reply = response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}. Is Ollama running?")

    return ChatResponse(reply=reply)


@app.post("/api/suggest")
async def suggest_analysis(session_id: str = Query(...), file_id: str = Query(...)):
    entry = _get_file_entry(session_id, file_id)
    summary = entry["summary"]
    context = _build_context_for_llm(summary)

    prompt = (
        f"Here is a summary of a data file named '{summary.get('filename', 'unknown')}':\n\n"
        f"{context}\n\n"
        f"Please suggest 3-5 specific, actionable data analysis ideas for this dataset. "
        f"For each suggestion, include:\n"
        f"1. A short title\n"
        f"2. A brief description of what to analyze and why it would be valuable\n\n"
        f"Format your response as a numbered list with clear, concise suggestions."
    )

    try:
        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {"role": "system", "content": "You are an expert data analyst. Provide clear, actionable analysis suggestions."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.5, "num_predict": 800},
        )
        reply = response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}. Is Ollama running?")

    return JSONResponse(content={"suggestions": reply})


def _build_eda_context(summary: dict, cleaning: dict) -> str:
    """Build an EDA + cleaning context string for the LLM."""
    parts = [
        f"File: {summary['filename']}",
        f"Rows: {summary['row_count']}, Columns: {summary['col_count']}",
        f"Numeric columns: {', '.join(summary.get('numeric_columns', []))}",
        f"Categorical columns: {', '.join(summary.get('categorical_columns', []))}",
        f"\nData Quality Issues ({cleaning['total_issues']} total):",
    ]
    for issue in cleaning["issues"]:
        parts.append(f"  [{issue['severity'].upper()}] {issue['type']}: {issue['detail']}")
    return "\n".join(parts)


def _build_context_for_llm(data: dict) -> str:
    parts = [f"Filename: {data['filename']}", f"Type: {data['type']}"]

    if data["type"] == "tabular":
        parts.append(f"Rows: {data['row_count']}, Columns: {data['col_count']}")
        columns = data["columns"][:30]
        if len(data["columns"]) > 30:
            parts.append(f"Column names (showing 30 of {data['col_count']}): {', '.join(columns)}")
        else:
            parts.append(f"Column names: {', '.join(columns)}")
        parts.append("\nColumn Statistics:")
        for col, stats in list(data.get("stats", {}).items())[:25]:
            stat_str = json.dumps(stats)
            if len(stat_str) > 300:
                stat_str = stat_str[:300] + "..."
            parts.append(f"  - {col}: {stat_str}")
        parts.append("\nFirst 10 rows (preview):")
        preview_cols = data["columns"][:15]
        for i, row in enumerate(data.get("preview_rows", [])[:10]):
            trimmed = {k: str(v)[:60] for k, v in row.items() if k in preview_cols}
            parts.append(f"  Row {i+1}: {json.dumps(trimmed)}")

    elif data["type"] == "text":
        parts.append(f"Total lines: {data.get('total_lines', '?')}")
        parts.append(f"Total characters: {data.get('total_chars', '?')}")
        parts.append(f"\nFirst 50 lines:\n{data.get('preview', '')}")

    elif data["type"] == "powerbi":
        parts.append(f"Archive entries: {data.get('total_entries', '?')}")
        parts.append(f"Data entries found: {data.get('data_entries', '?')}")
        parts.append(f"\nExtracted preview:\n{data.get('preview', '')}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Serve static frontend
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")