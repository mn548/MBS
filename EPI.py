# EPI
import pandas as pd
import numpy as np
import re

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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

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
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VAL_SPLIT, verbose=1)

probs = model.predict(X, batch_size=256, verbose=0)

# Continuous sentiment in [-1, +1]: (-1)*P(neg) + 0*P(neu) + (+1)*P(pos)
df["sentiment"] = (-1.0 * probs[:, 0]) + (0.0 * probs[:, 1]) + (1.0 * probs[:, 2])

# 3) LDA TOPIC MODEL (gensim)
from gensim import corpora, models

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
