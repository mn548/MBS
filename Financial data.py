# Financial data extraction for Marina Bay Sands quarterly metrics
import pandas as pd
import numpy as np
from pathlib import Path
import re

DATA_DIR = Path("/Users/monica/Desktop/MBS/Financial data")
OUTPUT_CSV = "mbs_quarterly_metrics.csv"
PROPERTY_NAME = "Marina Bay Sands"

METRIC_PATTERNS = {
    "occupancy": re.compile(r"occupancy", re.IGNORECASE),
    "revpar": re.compile(r"(revenue per available room|revpar)", re.IGNORECASE),
    "trevpar": re.compile(r"(total\s*revpar|trevpar)", re.IGNORECASE),
}

# Q1–Q3 only (as per your research design)
MONTH_TO_Q = {"March": "Q1", "June": "Q2", "September": "Q3"}

# Handle month abbreviations too (Jun vs June)
MONTH_ABBR = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
    "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"
}

def clean_number(x):
    """
    Parse numeric cells robustly.
    - Removes $ , % and whitespace
    - Converts (123) to -123
    - Returns float or NaN
    NOTE: Occupancy scaling (0.948 vs 94.8) is handled separately by normalise_metric().
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()

    if s in ["—", "-", "–", ""]:
        return np.nan

    # remove separators/currency/percent
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()

    # handle parentheses negatives
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    try:
        return float(s)
    except ValueError:
        return np.nan

def normalise_metric(metric_key: str, val: float) -> float:
    """
    Normalise inconsistent Excel formats across files.
    Occupancy can be stored as:
      - 0.948 (fraction) OR
      - 94.8 (percent)
    We convert to percent consistently (0–100 scale).
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val

    if metric_key == "occupancy":
        # If stored as a fraction (0–1), convert to percent.
        # Threshold 1.5 safely catches 0.xx formats but leaves 67.9 as-is.
        if val <= 1.5:
            return val * 100.0
        return val

    return val

def flatten_text(df: pd.DataFrame) -> str:
    text = " ".join(df.astype(str).values.flatten())
    return re.sub(r"\s+", " ", text)

def find_year_columns(df: pd.DataFrame) -> tuple[int, int] | None:
    """
    Find the two year header columns (e.g., 2022 and 2021).
    """
    top = df.head(60)
    hits = []
    for col in range(top.shape[1]):
        vals = pd.to_numeric(top[col], errors="coerce")
        years = vals[(vals >= 2000) & (vals <= 2100)].dropna().unique()
        for y in years:
            hits.append((int(y), col))

    if not hits:
        return None

    years = sorted({y for y, _ in hits}, reverse=True)
    if len(years) < 2:
        return None

    y_cur, y_prev = years[0], years[1]
    cur_cols = [c for y, c in hits if y == y_cur]
    prev_cols = [c for y, c in hits if y == y_prev]
    if not cur_cols or not prev_cols:
        return None

    return cur_cols[0], prev_cols[0]

def infer_quarter(df: pd.DataFrame) -> str | None:
    """
    Quarter inference robust to the screenshot structure:
    - 'Three Months Ended <Month> <Day>,' may NOT include the year in the same cell.
    Priority:
      1) Try parsing the 'Three Months Ended ...' month/day and attach year from header years.
      2) Fallback to 'Period End: Jun 30, 2022'
    Returns like '2022Q2' or None.
    """
    text = flatten_text(df)

    # 1) Three Months Ended (month/day); year may be separate
    m = re.search(r"three months ended\s+([A-Za-z]+)\s+(\d{1,2})(?:,)?", text, flags=re.IGNORECASE)
    if m:
        mon, _day = m.groups()
        mon = mon.strip().capitalize()
        mon = MONTH_ABBR.get(mon[:3].capitalize(), mon)
        q = MONTH_TO_Q.get(mon)
        if not q:
            return None

        # attach year from header (largest year found)
        top = df.head(60)
        years_found = pd.to_numeric(top.values.flatten(), errors="coerce")
        years_found = years_found[(years_found >= 2000) & (years_found <= 2100)]
        if len(years_found) == 0:
            return None
        year = int(np.nanmax(years_found))
        return f"{year}{q}"

    # 2) Fallback: Period End line includes full date
    m = re.search(r"Period End:\s*([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})", text, flags=re.IGNORECASE)
    if m:
        mon, _, year = m.groups()
        mon = mon.strip().capitalize()
        mon = MONTH_ABBR.get(mon[:3].capitalize(), mon)
        q = MONTH_TO_Q.get(mon)
        return f"{year}{q}" if q else None

    return None

def is_room_revenue_table(df: pd.DataFrame) -> bool:
    """
    Return True only if the table relates to hotel room operations.
    Excludes mall / retail tables that also contain 'Occupancy'.
    """
    text = flatten_text(df).lower()

    # Must mention room revenue or RevPAR
    if "room revenue" in text or "revpar" in text:
        return True

    return False

def extract_from_sheet(file_path: Path, sheet: str, debug: bool=False) -> dict | None:
    df = pd.read_excel(file_path, sheet_name=sheet, header=None, engine="xlrd")

# Skip mall / retail tables (e.g. "Mall revenues in millions")
    if not is_room_revenue_table(df):
        return None

    quarter = infer_quarter(df)
    if quarter is None:
        return None

    year_cols = find_year_columns(df)
    if year_cols is None:
        return None
    col_cur, col_prev = year_cols

    # Find Marina Bay Sands row in column 1
    col1 = df[1].astype(str)
    idxs = col1[col1.str.contains(PROPERTY_NAME, case=False, na=False)].index.tolist()
    if not idxs:
        return None
    i = idxs[0]

    # Only look at immediate metric rows below Marina Bay Sands to avoid overwriting with later sections
    block = df.loc[i+1:i+8, [1, col_cur]].copy()
    block.columns = ["metric_raw", "value_raw"]

    record = {"quarter": quarter, "source_file": file_path.name, "source_sheet": sheet}

    for _, r in block.iterrows():
        metric_text = str(r["metric_raw"]).strip()

        # Stop if we hit the next section header (e.g., 'U.S. Operations:')
        if metric_text.endswith(":") and "operations" in metric_text.lower():
            break

        for key, pat in METRIC_PATTERNS.items():
            # IMPORTANT: only set the first match; do not overwrite later
            if key not in record and pat.search(metric_text):
                raw_val = clean_number(r["value_raw"])
                record[key] = normalise_metric(key, raw_val)

    # Require at least RevPAR or occupancy
    if ("revpar" not in record) and ("occupancy" not in record):
        return None

    if debug:
        print(f"[MATCH] {file_path.name} | {sheet} | {quarter} | {record}")

    return record

def extract_all_files(debug: bool=False) -> pd.DataFrame:
    records = []
    for file in DATA_DIR.glob("*.xls"):
        try:
            xls = pd.ExcelFile(file, engine="xlrd")

            # Prefer TABLE## sheets
            sheets = [s for s in xls.sheet_names if re.match(r"TABLE\d+", str(s), flags=re.IGNORECASE)]
            if not sheets:
                sheets = xls.sheet_names

            for sheet in sheets:
                rec = extract_from_sheet(file, sheet, debug=debug)
                if rec:
                    records.append(rec)

        except Exception as e:
            print(f"Skipping {file.name} due to error: {e}")

    df = pd.DataFrame(records)
    if df.empty:
        print("No rows extracted.")
        print("Most likely causes:")
        print("  1) xlrd not installed in the SAME environment running the script")
        print("  2) Your key sheets are not named TABLE##")
        print("  3) Column 1 doesn't hold the property names in some files")
        return df

    # Deduplicate by quarter, keep the row with most filled metrics
    metric_cols = [c for c in ["occupancy", "revpar", "trevpar"] if c in df.columns]
    df["_filled"] = df[metric_cols].notna().sum(axis=1) if metric_cols else 0
    df = df.sort_values(["quarter", "_filled"], ascending=[True, False]).drop_duplicates("quarter", keep="first")
    df = df.drop(columns=["_filled"])

    # Sort quarters chronologically
    def qkey(q):
        m = re.match(r"(\d{4})Q([1-4])", str(q))
        return (int(m.group(1)), int(m.group(2))) if m else (9999, 9)

    df = df.sort_values("quarter", key=lambda s: s.map(qkey)).reset_index(drop=True)

    # Sanity check: occupancy should be 0–100 after normalisation
    if "occupancy" in df.columns:
        bad = df[(df["occupancy"] < 0) | (df["occupancy"] > 100)]
        if not bad.empty:
            print("\nWARNING: Occupancy outside 0–100 found (check formatting in these rows):")
            print(bad[["quarter", "occupancy", "source_file", "source_sheet"]])

    return df

if __name__ == "__main__":
    # debug=True prints matches (useful for checking one run)
    df = extract_all_files(debug=True)
    if not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved: {OUTPUT_CSV}")
    print(df)
