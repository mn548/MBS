# LOAD ALL NECESSARY LIBRARIES
import pandas as pd
import numpy as np

_ALIAS_MAP = {
    "complex_": np.complex128,
    "float_": np.float64,
    "string_": np.bytes_,
    "unicode_": np.str_,
}

for name, dtype in _ALIAS_MAP.items():
    if not hasattr(np, name):
        setattr(np, name, dtype)

for name, dtype in _ALIAS_MAP.items():
    if name not in np.sctypeDict:
        np.sctypeDict[name] = dtype

from pathlib import Path
import re
import statsmodels.api as sm
from docx import Document
from docx.shared import Inches

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from gensim import corpora, models

# FINANCIAL DATA PREPROCESSING
# Financial data extraction for Marina Bay Sands quarterly metrics
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

# EPI
# CONFIG

INPUT_CSV = "tripadvisor_mbs_review_from201501_v2 2.csv"
ENCODING = "latin1"

# GRU settings
MAX_WORDS = 20000
MAX_LEN = 200
EMBED_DIM = 128
GRU_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 5
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# LDA settings
N_TOPICS = 8
LDA_PASSES = 10
LDA_NO_BELOW = 20
LDA_NO_ABOVE = 0.5

# Outputs
OUT_EPI = "epi_quarterly_true.csv"
OUT_TOPICS = "epi_topics.csv"
OUT_ASPECT_Q = "epi_aspect_quarterly.csv"

# HELPERS
def parse_quarter(date_of_stay):
    """Convert date_of_stay like '2022/8' to pandas Period (quarterly)."""
    if pd.isna(date_of_stay):
        return pd.NaT
    dt = pd.to_datetime(str(date_of_stay) + "/01", errors="coerce")
    return dt.to_period("Q") if not pd.isna(dt) else pd.NaT

STOPWORDS = {
    "the","and","was","were","with","this","that","from","have","had","has",
    "for","not","are","but","you","your","they","their","them","his","her",
    "she","him","its","our","out","very","there","what","when","where","which",
    "who","will","would","could","should","into","about","after","before",
    "over","under","again","also","because","while","during","many","much"
}

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = s.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

def rating_to_label(r):
    """
    Convert 1–5 rating to 3-class sentiment labels:
      1–2 -> 0 (neg)
      3   -> 1 (neutral)
      4–5 -> 2 (pos)
    """
    r = float(r)
    if r <= 2:
        return 0
    elif r == 3:
        return 1
    else:
        return 2

# LOAD + FILTER + TOKENISE
df = pd.read_csv(INPUT_CSV, encoding=ENCODING)

df["quarter"] = df["date_of_stay"].map(parse_quarter)
df = df.dropna(subset=["quarter", "ratings"]).copy()

# Keep only 2015Q1 – 2022Q3, and only Q1–Q3
df = df[
    (df["quarter"] >= pd.Period("2015Q1", freq="Q")) &
    (df["quarter"] <= pd.Period("2022Q3", freq="Q")) &
    (df["quarter"].astype(str).str.endswith(("Q1", "Q2", "Q3")))
].copy()

# Text tokens
df["tokens"] = (df["title"].fillna("") + " " + df["content"].fillna("")).apply(clean_text)
df = df[df["tokens"].str.len() > 5].copy()

print("Quarter range:", df["quarter"].min(), "→", df["quarter"].max())
print("Unique quarters:", df["quarter"].nunique(), "(expected 24)")

# GRU SENTIMENT (tf.keras)
# Reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

df["label"] = df["ratings"].map(rating_to_label).astype(int)

texts = df["tokens"].apply(lambda x: " ".join(x))

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=MAX_LEN)

y = df["label"].values

model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
    GRU(GRU_UNITS),
    Dense(3, activation="softmax")
])

model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Ensure clean dtypes for TF
X = np.asarray(X, dtype=np.int32)
y = np.asarray(y, dtype=np.int32)

print("X:", type(X), X.dtype, X.shape)
print("y:", type(y), y.dtype, y.shape)

model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VAL_SPLIT, verbose=1)

probs = model.predict(X, batch_size=256, verbose=0)

# Continuous sentiment in [-1, +1]: (-1)*P(neg) + 0*P(neu) + (+1)*P(pos)
df["sentiment"] = (-1.0 * probs[:, 0]) + (0.0 * probs[:, 1]) + (1.0 * probs[:, 2])

# 3) LDA TOPIC MODEL (gensim)
dictionary = corpora.Dictionary(df["tokens"])
dictionary.filter_extremes(no_below=LDA_NO_BELOW, no_above=LDA_NO_ABOVE)

corpus = [dictionary.doc2bow(toks) for toks in df["tokens"]]

lda = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=N_TOPICS,
    random_state=RANDOM_SEED,
    passes=LDA_PASSES
)

def doc_topic_vector(bow):
    vec = np.zeros(N_TOPICS, dtype=float)
    for k, p in lda.get_document_topics(bow, minimum_probability=0.0):
        vec[k] = p
    # safety normalisation
    s = vec.sum()
    return vec / s if s > 0 else vec

df["theta"] = [doc_topic_vector(b) for b in corpus]

Theta = np.vstack(df["theta"].values)  # (N, K)
w_k = Theta.mean(axis=0)
w_k = w_k / w_k.sum()  # topic weights

# ASPECT-WEIGHTED EPI (QUARTERLY)
rows = []
for q, g in df.groupby("quarter"):
    Theta_q = np.vstack(g["theta"].values)        # (n_q, K)
    s_q = g["sentiment"].values                   # (n_q,)

    # aspect sentiment per topic in quarter: weighted by topic proportions
    denom = Theta_q.sum(axis=0) + 1e-9
    numer = (Theta_q * s_q[:, None]).sum(axis=0)
    S_kq = numer / denom

    # topic prevalence in quarter (useful for diagnostics)
    prev_kq = Theta_q.mean(axis=0)

    row = {"quarter": str(q), "n_reviews": int(len(g))}
    for k in range(N_TOPICS):
        row[f"topic_{k}_sent"] = float(S_kq[k])
        row[f"topic_{k}_prev"] = float(prev_kq[k])

    # EPI = sum_k w_k * S_kq
    row["EPI"] = float(S_kq @ w_k)
    rows.append(row)

aspect_q = pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)

# Ensure final window is exactly 2015Q1–2022Q3 and Q1–Q3 only
aspect_q["quarter_p"] = aspect_q["quarter"].apply(lambda x: pd.Period(x, freq="Q"))
aspect_q = aspect_q[
    (aspect_q["quarter_p"] >= pd.Period("2015Q1")) &
    (aspect_q["quarter_p"] <= pd.Period("2022Q3")) &
    (aspect_q["quarter"].str.endswith(("Q1","Q2","Q3")))
].copy()
aspect_q = aspect_q.drop(columns=["quarter_p"]).reset_index(drop=True)

epi_q = aspect_q[["quarter", "EPI", "n_reviews"]].copy()

print("Final EPI quarters:", len(epi_q), "(expected 24)")
print(epi_q.head())

# EXPORT TOPICS + DATASETS
topics = []
for k in range(N_TOPICS):
    words = [w for w, _ in lda.show_topic(k, topn=12)]
    topics.append({
        "topic": k,
        "keywords": ", ".join(words),
        "topic_weight_wk": float(w_k[k])
    })
topics_df = pd.DataFrame(topics)

epi_q.to_csv(OUT_EPI, index=False)
topics_df.to_csv(OUT_TOPICS, index=False)
aspect_q.to_csv(OUT_ASPECT_Q, index=False)

print("Saved:")
print(" -", OUT_EPI)
print(" -", OUT_TOPICS)
print(" -", OUT_ASPECT_Q)

# REGRESSION
# CONFIG (change paths if needed)
EPI_CSV = "epi_quarterly_true.csv"
FIN_CSV = "mbs_quarterly_metrics.csv"

OUT_MERGED = "mbs_merged_epi_financial.csv"
OUT_RESULTS = "regression_results_long.csv"
OUT_TABLE_DOCX = "apa_regression_table.docx"

DEP_VAR = "revpar"   # dependent variable in financial file

# HELPERS
def to_period_q(s: str) -> pd.Period:
    """Convert '2015Q1' -> Period('2015Q1', 'Q')"""
    return pd.Period(str(s), freq="Q")

def add_quarter_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonal control, avoiding dummy trap.

    IMPORTANT: The lagging removes Q1 observations (no q-1 because Q4 missing),
    so the regression sample typically contains only Q2 & Q3.
    In that case, if we include both Q2 and Q3 dummies + intercept, we get
    perfect collinearity because (Q2 + Q3) == 1 for every row.

    Solution: include ONLY one dummy, e.g. Q3 (Q2 is baseline).
    """
    qnum = df["quarter_p"].dt.quarter
    df["Q3"] = (qnum == 3).astype(int)
    return df

def debug_asarray(name, obj):
    import numpy as np
    print(f"\n{name}: type={type(obj)}")
    try:
        arr = np.asarray(obj)
        print(f"{name}: asarray -> {type(arr)}, dtype={getattr(arr,'dtype',None)}, shape={getattr(arr,'shape',None)}")
        return arr
    except Exception as e:
        print(f"{name}: asarray FAILED -> {e}")
        # if it's a pandas object, try extracting raw numpy
        try:
            arr = obj.to_numpy()
            print(f"{name}: to_numpy -> {type(arr)}, dtype={arr.dtype}, shape={arr.shape}")
            return arr
        except Exception as e2:
            print(f"{name}: to_numpy FAILED -> {e2}")
            raise

if "df_out" not in locals():
    df_out = pd.read_csv(OUT_MERGED)

    if "quarter_p" in df_out.columns:
        df_out["quarter_p"] = df_out["quarter_p"].apply(lambda x: pd.Period(x, freq="Q"))

assert "df_out" in locals(), "df_out not defined – check execution order"
reg = df_out.dropna(subset=["EPI_lag1", DEP_VAR]).copy()

X_base = reg[["EPI_lag1", "Q3"]].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(reg[DEP_VAR], errors="coerce")

print("reg shape:", reg.shape)
print(reg[["EPI_lag1", "Q3", DEP_VAR]].head())


X_base_np = debug_asarray("X_base", X_base.to_numpy(dtype=float))
y_np = debug_asarray("y", y.to_numpy(dtype=float))

def run_ols_hc3(y, X):
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X, missing="drop")
    res = model.fit(cov_type="HC3")
    return res

res_base = run_ols_hc3(y_np, X_base_np)

def format_p(p):
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}"

def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def apa_table_to_docx(res_list, model_names, filename):
    """
    Create a simple APA-style regression table in Word.
    Shows b (SE) with significance stars.
    """
    # Collect all coefficient names across models
    coef_names = set()
    for r in res_list:
        coef_names |= set(r.params.index)
    # Order: constant first, then EPI_lag1, then controls
    preferred_order = ["const", "EPI_lag1", "Q3", "occupancy", "trevpar"]
    remaining = [c for c in coef_names if c not in preferred_order]
    coef_order = [c for c in preferred_order if c in coef_names] + sorted(remaining)

    doc = Document()
    doc.add_heading("Regression Results (APA style)", level=1)

    # Table: rows = coefficients + model stats, cols = models+1
    table = doc.add_table(rows=1, cols=1 + len(res_list))
    hdr = table.rows[0].cells
    hdr[0].text = "Predictor"
    for j, name in enumerate(model_names, start=1):
        hdr[j].text = name

    # Coeff rows
    for c in coef_order:
        row = table.add_row().cells
        row[0].text = "Intercept" if c == "const" else c
        for j, r in enumerate(res_list, start=1):
            if c in r.params.index:
                b = r.params[c]
                se = r.bse[c]
                p = r.pvalues[c]
                row[j].text = f"{b:.3f} ({se:.3f}){stars(p)}"
            else:
                row[j].text = ""

    # Add model fit stats
    def add_stat(label, values):
        row = table.add_row().cells
        row[0].text = label
        for j, v in enumerate(values, start=1):
            row[j].text = v

    add_stat("N", [str(int(r.nobs)) for r in res_list])
    add_stat("R²", [f"{r.rsquared:.3f}" for r in res_list])
    add_stat("Adj. R²", [f"{r.rsquared_adj:.3f}" for r in res_list])
    add_stat("F (p)", [f"{r.fvalue:.2f} ({format_p(r.f_pvalue)})" if r.fvalue is not None else "" for r in res_list])

    doc.add_paragraph("\nNote. Entries are unstandardized coefficients with HC3 robust standard errors in parentheses.")
    doc.add_paragraph("Significance: * p < .05, ** p < .01, *** p < .001.")
    doc.save(filename)

# LOAD + MERGE
epi = pd.read_csv(EPI_CSV)
fin = pd.read_csv(FIN_CSV)

# Standardize quarter types
epi["quarter_p"] = epi["quarter"].apply(to_period_q)
fin["quarter_p"] = fin["quarter"].apply(to_period_q)

# Keep only necessary columns
epi = epi[["quarter", "quarter_p", "EPI", "n_reviews"]].copy()

# Financial file may have occupancy/trevpar missing -> keep if present
keep_fin_cols = ["quarter", "quarter_p", "revpar"]
for c in ["occupancy", "trevpar"]:
    if c in fin.columns:
        keep_fin_cols.append(c)

fin = fin[keep_fin_cols].copy()

# Merge on quarter
df = pd.merge(fin, epi, on=["quarter", "quarter_p"], how="inner")

# LAGGED EPI (true q-1)
# Create EPI_{q-1} by shifting period by +1 and merging back:
# For each quarter q, we want EPI from (q-1).
epi_lag = epi.copy()
epi_lag["quarter_p"] = epi_lag["quarter_p"] + 1  # move forward one quarter
epi_lag = epi_lag.rename(columns={"EPI": "EPI_lag1", "n_reviews": "n_reviews_lag1"})

df = pd.merge(df, epi_lag[["quarter_p", "EPI_lag1", "n_reviews_lag1"]], on="quarter_p", how="left")

# Add seasonal dummy (Q3 only to avoid dummy trap)
df = add_quarter_dummies(df)

# Save merged dataset
df_out = df.sort_values("quarter_p").copy()
df_out.to_csv(OUT_MERGED, index=False)
print(f"Saved merged dataset: {OUT_MERGED}")

# REGRESSIONS
# Drop rows without true lag
reg = df_out.dropna(subset=["EPI_lag1", DEP_VAR]).copy()

# Baseline model: RevPAR ~ EPI_lag1 + Q3
X_base = reg[["EPI_lag1", "Q3"]].copy()
y = reg[DEP_VAR].astype(float)
res_base = run_ols_hc3(y, X_base)

# Extended model: add occupancy + trevpar if available
X_ext_cols = ["EPI_lag1", "Q3"]
if "occupancy" in reg.columns:
    X_ext_cols.append("occupancy")
if "trevpar" in reg.columns:
    X_ext_cols.append("trevpar")

X_ext = reg[X_ext_cols].copy()
res_ext = run_ols_hc3(y, X_ext)

print("\nBaseline model summary (HC3):")
print(res_base.summary())
print("\nExtended model summary (HC3):")
print(res_ext.summary())

# EXPORT RESULTS (CSV + APA DOCX)
def results_to_long(res, model_name):
    rows = []
    for name in res.params.index:
        rows.append({
            "model": model_name,
            "term": name,
            "b": res.params[name],
            "se_hc3": res.bse[name],
            "t": res.tvalues[name],
            "p": res.pvalues[name],
        })
    # model stats
    rows.append({"model": model_name, "term": "N", "b": res.nobs, "se_hc3": np.nan, "t": np.nan, "p": np.nan})
    rows.append({"model": model_name, "term": "R2", "b": res.rsquared, "se_hc3": np.nan, "t": np.nan, "p": np.nan})
    rows.append({"model": model_name, "term": "Adj_R2", "b": res.rsquared_adj, "se_hc3": np.nan, "t": np.nan, "p": np.nan})
    return rows

long = pd.DataFrame(results_to_long(res_base, "Baseline") + results_to_long(res_ext, "Extended"))
long.to_csv(OUT_RESULTS, index=False)
print(f"Saved regression results (long CSV): {OUT_RESULTS}")

apa_table_to_docx([res_base, res_ext], ["Model 1 (Baseline)", "Model 2 (Extended)"], OUT_TABLE_DOCX)
print(f"Saved APA-style regression table: {OUT_TABLE_DOCX}")

# SANITY PRINTS
print("\nRegression sample quarters used:")
print(reg[["quarter", "EPI_lag1", DEP_VAR]].head(10))
print("\nN baseline:", int(res_base.nobs), " | N extended:", int(res_ext.nobs))
print("Note: Q1 observations drop out because Q4 is not available to create a true q−1 lag.")
