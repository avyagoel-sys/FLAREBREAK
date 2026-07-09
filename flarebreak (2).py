"""
=====================================================================
FLAREBREAK
An AI Early-Warning System for Predicting Dangerous Solar Radiation
Storms, Using Historical Solar Flare Data + Live NOAA Proton Flux
=====================================================================

HOW TO RUN THIS (Colab / Kaggle / Jupyter):
  1. Paste this ENTIRE file into ONE notebook cell.
  2. Run that cell (Shift+Enter). It will:
       - install nothing itself (see pip install line below, run once)
       - train the model
       - print results and save charts
       - leave `MODEL` and `FEATURE_COLUMNS` ready to use
  3. In a NEW cell, predict a brand new region like this:

        predict_new_region({
            "activity": 1,
            "evolution": 3,
            "prev_24hr_activity": 3,
            "area": 2,
        })

     That's it - one function, one dictionary, no setup needed.

HOW TO RUN THIS (plain Python / terminal):
  pip install ucimlrepo scikit-learn pandas matplotlib requests
  python flarebreak.py

DATASET:
  UCI Machine Learning Repository - Solar Flare Dataset
  https://archive.ics.uci.edu/dataset/89/solar+flare
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------
# THESE GLOBALS ARE FILLED IN WHEN THE SCRIPT RUNS. AFTER RUNNING, YOU
# CAN USE MODEL / FEATURE_COLUMNS / X / Y DIRECTLY IN YOUR NOTEBOOK, OR
# JUST CALL predict_new_region({...}) - SEE BOTTOM OF THIS FILE.
# ---------------------------------------------------------------------
MODEL = None
FEATURE_COLUMNS = None
X = None
Y = None
DECISION_THRESHOLD = 0.5  # will be tuned automatically during training


# =====================================================================
# PART A: HISTORICAL SOLAR FLARE MODEL (UCI dataset)
# =====================================================================
def load_solar_flare_data():
    """
    Tries to download the REAL UCI Solar Flare dataset.
    Falls back to synthetic placeholder data (same structure) only if
    there is no internet connection available.
    """
    column_names = [
        "zurich_class", "largest_spot_size", "spot_distribution",
        "activity", "evolution", "prev_24hr_activity",
        "historically_complex", "became_historically_complex",
        "area", "area_largest_spot",
        "c_class_count", "m_class_count", "x_class_count"
    ]

    # ---- Attempt 1: official ucimlrepo package ----
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=89)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df.columns = column_names
        print(f"[OK] Loaded REAL UCI Solar Flare dataset "
              f"({len(df)} active-region records).")
        return df, True
    except Exception:
        pass

    # ---- Attempt 2: raw CSV mirror on GitHub (same dataset, re-hosted) ----
    try:
        url = ("https://raw.githubusercontent.com/"
               "Ahmad-Alaziz/Solar-Flare-Detection-AI/main/data/flare.data2")
        df = pd.read_csv(url, sep=r"\s+", skiprows=1, header=None,
                          names=column_names)
        print(f"[OK] Loaded REAL UCI Solar Flare dataset via mirror "
              f"({len(df)} records).")
        return df, True
    except Exception:
        pass

    # ---- Fallback: synthetic data (SAME SCHEMA, clearly labeled) ----
    print("[WARNING] No internet connection detected. Using SYNTHETIC "
          "placeholder data with the same structure as the real UCI "
          "dataset, purely so the pipeline can be demonstrated.\n"
          "          -> Re-run with an internet connection for real data.")
    n = 1000
    df = pd.DataFrame({
        "zurich_class": np.random.choice(list("ABCDEFH"), n),
        "largest_spot_size": np.random.choice(list("XRSAHK"), n),
        "spot_distribution": np.random.choice(list("XOIC"), n),
        "activity": np.random.choice([1, 2], n),
        "evolution": np.random.choice([1, 2, 3], n),
        "prev_24hr_activity": np.random.choice([1, 2, 3], n),
        "historically_complex": np.random.choice([1, 2], n),
        "became_historically_complex": np.random.choice([1, 2], n),
        "area": np.random.choice([1, 2], n),
        "area_largest_spot": np.random.choice([1, 2], n),
    })
    risk_score = (df["activity"] == 1).astype(int) + (df["evolution"] == 3).astype(int) \
                 + (df["prev_24hr_activity"] == 3).astype(int)
    df["c_class_count"] = np.random.poisson(1 + risk_score)
    df["m_class_count"] = np.random.poisson(0.3 * risk_score)
    df["x_class_count"] = np.random.poisson(0.05 * risk_score)
    return df, False


def preprocess(df):
    """
    Encodes categorical columns and engineers the target label:
    DANGEROUS (1) if the region produced an M-class or X-class flare,
    SAFE (0) otherwise.
    """
    df = df.copy()
    categorical_cols = ["zurich_class", "largest_spot_size", "spot_distribution"]
    df = pd.get_dummies(df, columns=categorical_cols)

    df["dangerous_flare"] = (
        (df["m_class_count"] > 0) | (df["x_class_count"] > 0)
    ).astype(int)

    feature_cols = [c for c in df.columns if c not in
                    ["c_class_count", "m_class_count", "x_class_count", "dangerous_flare"]]

    return df[feature_cols], df["dangerous_flare"]


def train_model(X_train, y_train):
    """
    Trains the flare-risk classifier using two techniques that genuinely
    improve detection of the rare "Dangerous" class (not just raw
    accuracy, which is misleading on imbalanced data like this):

      1. SMOTE oversampling: creates synthetic extra examples of the
         rare Dangerous class during training, so the model sees a
         more balanced picture of what "dangerous" looks like.
      2. Decision threshold tuning: instead of the default 50% cutoff,
         we find the probability cutoff that gives the best balance
         of precision and recall, tuned on a separate VALIDATION split
         carved out of the training data only (never the test set,
         which would be data leakage and would fake the results).
    """
    global DECISION_THRESHOLD

    # Carve a validation set out of the TRAINING data only
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    # Oversample the rare "Dangerous" class in the training portion only
    n_minority = y_tr.sum()
    k_neighbors = min(5, max(1, n_minority - 1))
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_tr, y_tr)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_SEED
    )
    model.fit(X_resampled, y_resampled)

    # Tune the decision threshold on the VALIDATION set (not test set)
    val_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    if len(thresholds) > 0:
        best_idx = np.argmax(f1_scores[:-1])
        DECISION_THRESHOLD = float(thresholds[best_idx])
    else:
        DECISION_THRESHOLD = 0.5

    print(f"[Tuning] Best decision threshold found: {DECISION_THRESHOLD:.3f} "
          f"(instead of the default 0.5)")

    return model


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Decision threshold used: {DECISION_THRESHOLD:.3f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Dangerous"], zero_division=0))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    print(f"--> Dangerous-class Recall: {rec:.3f}  "
          f"(this is the number that matters most for an early-warning "
          f"system - it's the fraction of REAL dangerous flares the "
          f"model actually caught)")
    print(f"--> Dangerous-class Precision: {prec:.3f}  "
          f"(of everything flagged as dangerous, how many actually were)")

    if len(set(y_test)) > 1:
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.3f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Dangerous"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("FlareBreak: Confusion Matrix")
    plt.tight_layout()
    plt.savefig("flarebreak_confusion_matrix.png", dpi=150)
    plt.show()
    plt.close()
    print("[Saved] flarebreak_confusion_matrix.png")


def plot_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(10)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#0B3D91")
    plt.title("FlareBreak: Top 10 Most Important Predictive Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("flarebreak_feature_importance.png", dpi=150)
    plt.show()
    plt.close()
    print("[Saved] flarebreak_feature_importance.png")


def predict_new_region(region_features: dict, threshold=None):
    """
    *** THIS IS THE FUNCTION YOU USE TO TEST NEW DATA ***

    Give it a dictionary describing a solar active region, and it
    tells you the model's predicted risk of a dangerous flare.
    Uses the automatically-tuned DECISION_THRESHOLD by default (found
    during training to best balance catching real danger vs. false
    alarms) unless you explicitly pass a different one.
    """
    if threshold is None:
        threshold = DECISION_THRESHOLD

    if MODEL is None or FEATURE_COLUMNS is None:
        print("[ERROR] The model hasn't been trained yet. Run this script "
              "(or this notebook cell) fully first, then try again.")
        return None, None

    row = pd.DataFrame([{col: region_features.get(col, 0) for col in FEATURE_COLUMNS}])
    risk = MODEL.predict_proba(row)[0, 1]

    if risk >= threshold:
        alert = (f"[ALERT] Elevated radiation storm risk detected "
                  f"({risk*100:.1f}% probability). Recommend astronauts "
                  f"shelter or delay scheduled spacewalks.")
    else:
        alert = (f"[CLEAR] Low radiation storm risk "
                  f"({risk*100:.1f}% probability). Normal operations "
                  f"can proceed.")

    print(alert)
    return risk, alert


# =====================================================================
# PART B: NOAA GOES PROTON FLUX MODULE (live/recent real-world data)
# =====================================================================
#
# WHY THIS IS SEPARATE FROM PART A:
#   Part A (UCI) is a table of individual active regions from 1988 -
#   no timestamps, no shared ID with live data.
#   Part B is a continuous, real-time stream of actual proton flux
#   measurements from NOAA satellites right now.
#   They can't be honestly merged row-by-row, so FlareBreak treats
#   them as two complementary layers:
#     Part A -> "Is this TYPE of region historically risky?"
#     Part B -> "Is a storm actively happening RIGHT NOW?"
#
NOAA_PROTON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json"

S_SCALE_THRESHOLDS = [
    (100000, "S5 - Extreme"),
    (10000,  "S4 - Severe"),
    (1000,   "S3 - Strong"),
    (100,    "S2 - Moderate"),
    (10,     "S1 - Minor"),
]


def fetch_noaa_proton_flux(url=NOAA_PROTON_URL, timeout=15):
    """Downloads recent real proton flux measurements from NOAA SWPC."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        df = pd.DataFrame(response.json())

        energy_col = next((c for c in df.columns if "energy" in c.lower()), None)
        flux_col = next((c for c in df.columns if "flux" in c.lower()), None)
        time_col = next((c for c in df.columns if "time" in c.lower()), None)
        if not (energy_col and flux_col and time_col):
            raise ValueError("Unexpected NOAA JSON format.")

        df = df.rename(columns={energy_col: "energy", flux_col: "flux", time_col: "time_tag"})
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        df["flux"] = pd.to_numeric(df["flux"], errors="coerce")

        df_10mev = df[df["energy"].astype(str).str.contains("10 MeV", na=False)].copy()
        df_10mev = df_10mev.sort_values("time_tag").reset_index(drop=True)
        if df_10mev.empty:
            raise ValueError("No >=10 MeV proton flux records found.")

        print(f"[OK] Loaded REAL NOAA proton flux data ({len(df_10mev)} measurements).")
        return df_10mev
    except Exception as e:
        print(f"[WARNING] Could not fetch live NOAA data ({e}). "
              f"Using SYNTHETIC placeholder flux data instead.")
        return None


def make_synthetic_proton_flux(n=500):
    """Synthetic flux time series (same shape as real feed) for offline demo."""
    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="5min")
    baseline = np.random.uniform(0.05, 0.5, n)
    spike_start = int(n * 0.85)
    spike = np.zeros(n)
    spike[spike_start:spike_start + 20] = np.linspace(5, 150, 20)
    return pd.DataFrame({"time_tag": times, "energy": ">=10 MeV", "flux": baseline + spike})


def engineer_flux_features(df):
    """Builds rolling mean / rate-of-change / lag features and a next-step storm label."""
    df = df.copy().sort_values("time_tag").reset_index(drop=True)
    df["flux_rolling_mean"] = df["flux"].rolling(window=3, min_periods=1).mean()
    df["flux_rate_of_change"] = df["flux"].diff().fillna(0)
    df["flux_lag1"] = df["flux"].shift(1).fillna(df["flux"].iloc[0])
    df["flux_lag2"] = df["flux"].shift(2).fillna(df["flux"].iloc[0])
    df["storm_next_step"] = (df["flux"].shift(-1) >= 10).astype(int)
    return df.iloc[:-1]


def train_flux_model(df):
    """Trains a next-step storm predictor using a chronological train/test split."""
    feature_cols = ["flux", "flux_rolling_mean", "flux_rate_of_change", "flux_lag1", "flux_lag2"]
    X_flux, y_flux = df[feature_cols], df["storm_next_step"]

    split = int(len(df) * 0.8)
    X_train, X_test = X_flux.iloc[:split], X_flux.iloc[split:]
    y_train, y_test = y_flux.iloc[:split], y_flux.iloc[split:]

    if y_train.nunique() < 2:
        print("[INFO] No storm events in this training window - falling back "
              "to threshold-based monitoring only.")
        return None, feature_cols

    model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                    class_weight="balanced", random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    if len(X_test) > 0 and y_test.nunique() >= 1:
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"[Flux model] Next-step storm prediction accuracy: {acc:.3f}")

    return model, feature_cols


def current_storm_status(latest_flux_value):
    """Checks a flux value against the official NOAA S-scale thresholds."""
    for threshold, label in S_SCALE_THRESHOLDS:
        if latest_flux_value >= threshold:
            return label
    return "Below S1 (no radiation storm in progress)"


def plot_proton_flux(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["time_tag"], df["flux"], color="#0B3D91", linewidth=1)
    plt.axhline(10, color="orange", linestyle="--", label="S1 threshold (10 pfu)")
    plt.axhline(100, color="red", linestyle="--", label="S2 threshold (100 pfu)")
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Proton flux, >=10 MeV (pfu, log scale)")
    plt.title("FlareBreak: Recent Proton Flux (NOAA GOES)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("flarebreak_proton_flux.png", dpi=150)
    plt.show()
    plt.close()
    print("[Saved] flarebreak_proton_flux.png")


def run_flux_monitor():
    print("\n" + "=" * 60)
    print("PART B: NOAA PROTON FLUX EARLY-WARNING MONITOR")
    print("=" * 60)

    df = fetch_noaa_proton_flux()
    using_real = df is not None
    if df is None:
        df = make_synthetic_proton_flux()

    df = engineer_flux_features(df)
    plot_proton_flux(df)
    flux_model, feature_cols = train_flux_model(df)

    latest = df.iloc[-1]
    print(f"\nLatest measured proton flux (>=10 MeV): {latest['flux']:.2f} pfu")
    print(f"Current NOAA S-scale status: {current_storm_status(latest['flux'])}")

    if flux_model is not None:
        risk = flux_model.predict_proba(latest[feature_cols].to_frame().T)[0, 1]
        print(f"Model's predicted probability of storm at NEXT reading: {risk*100:.1f}%")

    if not using_real:
        print("\nReminder: this used SYNTHETIC placeholder proton flux data "
              "because no internet connection was available.")


# =====================================================================
# RUN EVERYTHING (this executes automatically whether you run this as
# a script with `python flarebreak.py`, OR paste it into a Colab/
# Jupyter cell and run the cell directly)
# =====================================================================
print("Starting FlareBreak pipeline...\n")

df, using_real_data = load_solar_flare_data()
X, Y = preprocess(df)
print(f"\nDangerous flare rate in dataset: {Y.mean()*100:.1f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=RANDOM_SEED, stratify=Y
)

MODEL = train_model(X_train, y_train)
FEATURE_COLUMNS = list(X.columns)

evaluate_model(MODEL, X_test, y_test)
plot_feature_importance(MODEL, FEATURE_COLUMNS)

print("\n" + "=" * 60)
print("EARLY WARNING SYSTEM DEMO")
print("=" * 60)
predict_new_region(X_test.iloc[0].to_dict())

if not using_real_data:
    print("\nReminder: this run used SYNTHETIC placeholder data because "
          "no internet connection was available.")

run_flux_monitor()

print("\nDone. MODEL and FEATURE_COLUMNS are ready to use.")

# =====================================================================
# >>> TEST YOUR OWN REGION HERE <<<
# Edit the numbers below to describe any region you want to check,
# then just re-run this ENTIRE cell (Ctrl+Enter / Shift+Enter).
# You do NOT need a separate cell - this runs automatically below.
# =====================================================================
print("\n" + "=" * 60)
print("CUSTOM REGION TEST")
print("=" * 60)

my_region = {
    "activity": 1,                    # 1 = reduced, 2 = unchanged
    "evolution": 3,                   # 1 = decay, 2 = no growth, 3 = growth
    "prev_24hr_activity": 3,          # 1 = nothing, 2 = one M1, 3 = more than one M1
    "area": 2,                        # 1 = small, 2 = large
}

predict_new_region(my_region)

