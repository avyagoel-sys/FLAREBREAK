"""
=====================================================================
FLAREBREAK
An AI Early-Warning System for Predicting Dangerous Solar Radiation
Storms, Using Historical Solar Flare Data
=====================================================================

WHAT THIS SCRIPT DOES (in plain English):
  1. Loads real historical solar activity data (UCI Solar Flare Dataset)
  2. Cleans and prepares it for machine learning
  3. Trains a Random Forest model to predict whether a solar active
     region will produce a DANGEROUS flare (M-class or X-class) in
     the next 24 hours
  4. Tests how accurate the model is on data it has never seen
  5. Shows which solar features matter most for prediction
  6. Demonstrates a simple "early warning" alert function

DATASET:
  UCI Machine Learning Repository - Solar Flare Dataset
  https://archive.ics.uci.edu/dataset/89/solar+flare

  Each row = one solar active region.
  Columns 1-10 = properties of that region (size, complexity, etc.)
  Columns 11-13 = how many C-class, M-class, and X-class flares
                   that region produced in the next 24 hours.

HOW TO RUN:
  1. Install requirements:
       pip install ucimlrepo scikit-learn pandas matplotlib --break-system-packages
  2. Run:
       python flarebreak.py
  3. Output charts will be saved in the same folder as this script.
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
    ConfusionMatrixDisplay, roc_auc_score
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =====================================================================
# STEP 1: LOAD THE DATA
# =====================================================================
def load_solar_flare_data():
    """
    Tries to download the REAL UCI Solar Flare dataset.
    If there is no internet connection available, it falls back to a
    small synthetic dataset with the exact same structure, so the rest
    of the pipeline can still be demonstrated end-to-end.
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
        return df, real_data_flag(True)
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
        return df, real_data_flag(True)
    except Exception:
        pass

    # ---- Fallback: synthetic data (SAME SCHEMA, clearly labeled) ----
    print("[WARNING] No internet connection detected. Using SYNTHETIC "
          "placeholder data with the same structure as the real UCI "
          "dataset, purely so the pipeline can be demonstrated.\n"
          "          -> Run this script again with an internet connection "
          "to train on real historical data.")
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
    # Simulate flare counts loosely correlated with activity/evolution
    risk_score = (df["activity"] == 1).astype(int) + (df["evolution"] == 3).astype(int) \
                 + (df["prev_24hr_activity"] == 3).astype(int)
    df["c_class_count"] = np.random.poisson(1 + risk_score)
    df["m_class_count"] = np.random.poisson(0.3 * risk_score)
    df["x_class_count"] = np.random.poisson(0.05 * risk_score)
    return df, real_data_flag(False)


def real_data_flag(is_real):
    return is_real


# =====================================================================
# STEP 2: PREPROCESS THE DATA
# =====================================================================
def preprocess(df):
    """
    - Encodes categorical columns (letters -> numbers) so the model
      can use them.
    - Engineers the target label: DANGEROUS (1) if the region produced
      at least one M-class or X-class flare, SAFE (0) otherwise.
      (M and X class flares are the ones strong enough to meaningfully
      raise astronaut radiation exposure.)
    """
    df = df.copy()

    categorical_cols = ["zurich_class", "largest_spot_size", "spot_distribution"]
    df = pd.get_dummies(df, columns=categorical_cols)

    df["dangerous_flare"] = (
        (df["m_class_count"] > 0) | (df["x_class_count"] > 0)
    ).astype(int)

    feature_cols = [c for c in df.columns if c not in
                    ["c_class_count", "m_class_count", "x_class_count", "dangerous_flare"]]

    X = df[feature_cols]
    y = df["dangerous_flare"]
    return X, y


# =====================================================================
# STEP 3: TRAIN THE MODEL
# =====================================================================
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",   # dangerous flares are rare -> balance classes
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    return model


# =====================================================================
# STEP 4: EVALUATE THE MODEL
# =====================================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {acc:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=["Safe", "Dangerous"]))

    # Only compute AUC if both classes are present
    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score: {auc:.3f}")

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Safe", "Dangerous"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("FlareBreak: Confusion Matrix")
    plt.tight_layout()
    plt.savefig("flarebreak_confusion_matrix.png", dpi=150)
    plt.close()
    print("\n[Saved] flarebreak_confusion_matrix.png")

    return y_pred, y_proba


# =====================================================================
# STEP 5: FEATURE IMPORTANCE (which factors matter most?)
# =====================================================================
def plot_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(10)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#0B3D91")
    plt.title("FlareBreak: Top 10 Most Important Predictive Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("flarebreak_feature_importance.png", dpi=150)
    plt.close()
    print("[Saved] flarebreak_feature_importance.png")


# =====================================================================
# STEP 6: EARLY WARNING FUNCTION (the actual "product")
# =====================================================================
def early_warning_alert(model, feature_cols, region_features: dict, threshold=0.5):
    """
    Takes a dictionary of a single active region's features and returns
    a risk score + plain-English alert, just like a real early-warning
    system would.

    Example:
        early_warning_alert(model, feature_cols, {
            "activity": 1, "evolution": 3, "prev_24hr_activity": 3, ...
        })
    """
    row = pd.DataFrame([{col: region_features.get(col, 0) for col in feature_cols}])
    risk = model.predict_proba(row)[0, 1]

    if risk >= threshold:
        alert = (f"[ALERT] Elevated radiation storm risk detected "
                 f"({risk*100:.1f}% probability). Recommend astronauts "
                 f"shelter or delay scheduled spacewalks.")
    else:
        alert = (f"[CLEAR] Low radiation storm risk "
                 f"({risk*100:.1f}% probability). Normal operations "
                 f"can proceed.")
    return risk, alert


# =====================================================================
# PART B: NOAA GOES PROTON FLUX MODULE
# (This is the second dataset promised in the project's dataset
#  submission: real-time / recent historical proton flux data.)
# =====================================================================
#
# NOTE ON WHY THIS IS A SEPARATE MODEL, NOT MERGED ROW-BY-ROW WITH PART A:
#   The UCI dataset (Part A) is a table of individual solar ACTIVE REGIONS
#   from 1988, each with a single "did it flare" outcome.
#   The NOAA proton flux feed (Part B) is a continuous TIME SERIES of
#   real particle measurements, updated every few minutes.
#   These two do not share a row-level key (no shared region ID or
#   timestamp), so they cannot be honestly joined into one table.
#   Instead, FlareBreak uses them as two complementary layers:
#     Part A -> "Is this type of active region historically risky?"
#     Part B -> "Is a radiation storm actively happening right now,
#                based on the trend in real proton flux?"
#
NOAA_PROTON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json"

# Official NOAA Solar Radiation Storm (S-scale) thresholds, in proton
# flux units (pfu), for the >=10 MeV channel:
S_SCALE_THRESHOLDS = [
    (100000, "S5 - Extreme"),
    (10000,  "S4 - Severe"),
    (1000,   "S3 - Strong"),
    (100,    "S2 - Moderate"),
    (10,     "S1 - Minor"),
]


def fetch_noaa_proton_flux(url=NOAA_PROTON_URL, timeout=15):
    """
    Downloads recent (rolling 7-day) real proton flux measurements from
    NOAA's Space Weather Prediction Center.

    Returns a tidy DataFrame with columns: time_tag, energy, flux
    filtered to the >=10 MeV channel (the channel that defines the
    NOAA S-scale radiation storm thresholds), or None if the data
    could not be retrieved (e.g. no internet connection).
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        raw = response.json()
        df = pd.DataFrame(raw)

        # Different NOAA feeds use slightly different key names over
        # time, so we check for the common variants defensively.
        energy_col = next((c for c in df.columns if "energy" in c.lower()), None)
        flux_col = next((c for c in df.columns if "flux" in c.lower()), None)
        time_col = next((c for c in df.columns if "time" in c.lower()), None)

        if not (energy_col and flux_col and time_col):
            raise ValueError("Unexpected NOAA JSON format - required "
                              "columns not found.")

        df = df.rename(columns={energy_col: "energy",
                                 flux_col: "flux",
                                 time_col: "time_tag"})
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        df["flux"] = pd.to_numeric(df["flux"], errors="coerce")

        # Keep only the >=10 MeV channel (this is the channel the
        # official NOAA S-scale is based on).
        df_10mev = df[df["energy"].astype(str).str.contains("10 MeV", na=False)].copy()
        df_10mev = df_10mev.sort_values("time_tag").reset_index(drop=True)

        if df_10mev.empty:
            raise ValueError("No >=10 MeV proton flux records found in feed.")

        print(f"[OK] Loaded REAL NOAA proton flux data "
              f"({len(df_10mev)} measurements over the last ~7 days).")
        return df_10mev

    except Exception as e:
        print(f"[WARNING] Could not fetch live NOAA proton flux data "
              f"({e}). Using SYNTHETIC placeholder flux data instead.\n"
              f"          -> Run this script again with an internet "
              f"connection to use real, current NOAA data.")
        return None


def make_synthetic_proton_flux(n=500):
    """
    Generates a synthetic proton flux time series with the same shape
    as the real NOAA feed, purely so the pipeline can be demonstrated
    without an internet connection. Occasionally injects a simulated
    'storm spike' so the alert logic has something to detect.
    """
    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="5min")
    baseline = np.random.uniform(0.05, 0.5, n)  # quiet background flux (pfu)

    # Inject one simulated storm spike near the end of the series
    spike_start = int(n * 0.85)
    spike = np.zeros(n)
    spike[spike_start:spike_start + 20] = np.linspace(5, 150, 20)
    flux = baseline + spike

    df = pd.DataFrame({
        "time_tag": times,
        "energy": ">=10 MeV",
        "flux": flux,
    })
    return df


def engineer_flux_features(df):
    """
    Builds simple time-series features from past flux readings:
      - rolling mean (smoothed recent trend)
      - rate of change from the previous reading
      - a label: did the NEXT reading cross the S1 storm threshold (10 pfu)?
    This label is what lets us train a model to predict a storm
    slightly BEFORE it's officially confirmed, using only past data.
    """
    df = df.copy().sort_values("time_tag").reset_index(drop=True)
    df["flux_rolling_mean"] = df["flux"].rolling(window=3, min_periods=1).mean()
    df["flux_rate_of_change"] = df["flux"].diff().fillna(0)
    df["flux_lag1"] = df["flux"].shift(1).fillna(df["flux"].iloc[0])
    df["flux_lag2"] = df["flux"].shift(2).fillna(df["flux"].iloc[0])

    # Label: will the NEXT reading be a storm (>=10 pfu, NOAA S1 threshold)?
    df["storm_next_step"] = (df["flux"].shift(-1) >= 10).astype(int)
    df = df.iloc[:-1]  # drop last row (no "next" reading to label)
    return df


def train_flux_model(df):
    """
    Trains a simple model on the engineered flux features to predict
    whether the NEXT reading will cross the storm threshold.
    Uses a chronological (not random) train/test split, since this is
    time-series data and the model should only ever learn from the past
    to predict the future - never the reverse.
    """
    feature_cols = ["flux", "flux_rolling_mean", "flux_rate_of_change",
                     "flux_lag1", "flux_lag2"]
    X = df[feature_cols]
    y = df["storm_next_step"]

    split_point = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    if y_train.nunique() < 2:
        print("[INFO] No storm events in this training window - the "
              "recent period was quiet. Skipping model training and "
              "falling back to threshold-based monitoring only.")
        return None, feature_cols

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        class_weight="balanced", random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    if len(X_test) > 0 and y_test.nunique() >= 1:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"[Flux model] Next-step storm prediction accuracy on "
              f"held-out recent data: {acc:.3f}")

    return model, feature_cols


def current_storm_status(latest_flux_value):
    """
    Rule-based check against the OFFICIAL NOAA S-scale thresholds.
    This always works, regardless of how much training data is
    available, and reflects real operational NOAA criteria.
    """
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
    plt.close()
    print("[Saved] flarebreak_proton_flux.png")


def run_flux_monitor():
    """
    Runs the full NOAA proton flux module end-to-end:
    fetch -> engineer features -> train -> report current status.
    """
    print("\n" + "=" * 60)
    print("PART B: NOAA PROTON FLUX EARLY-WARNING MONITOR")
    print("=" * 60)

    df = fetch_noaa_proton_flux()
    using_real_flux_data = df is not None
    if df is None:
        df = make_synthetic_proton_flux()

    df = engineer_flux_features(df)
    plot_proton_flux(df)

    model, feature_cols = train_flux_model(df)

    latest = df.iloc[-1]
    status = current_storm_status(latest["flux"])
    print(f"\nLatest measured proton flux (>=10 MeV): {latest['flux']:.2f} pfu")
    print(f"Current NOAA S-scale status: {status}")

    if model is not None:
        next_step_risk = model.predict_proba(
            latest[feature_cols].to_frame().T
        )[0, 1]
        print(f"Model's predicted probability of storm at NEXT reading: "
              f"{next_step_risk*100:.1f}%")

    if not using_real_flux_data:
        print("\nReminder: this run used SYNTHETIC placeholder proton flux "
              "data because no internet connection was available.")



def main():
    print("Starting FlareBreak pipeline...\n")

    df, using_real_data = load_solar_flare_data()

    X, y = preprocess(df)
    print(f"\nDangerous flare rate in dataset: {y.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X.columns)

    # ---- Demo: run the early warning system on one example region ----
    print("\n" + "=" * 60)
    print("EARLY WARNING SYSTEM DEMO")
    print("=" * 60)

    example_region = X_test.iloc[0].to_dict()
    risk, alert = early_warning_alert(model, X.columns, example_region)
    print(alert)

    if not using_real_data:
        print("\nReminder: this run used SYNTHETIC placeholder data because "
              "no internet connection was available. Re-run with an "
              "internet connection to train on the real historical dataset.")

    # ---- Part B: real-time NOAA proton flux monitoring ----
    run_flux_monitor()

    print("\nDone. Charts saved in the current folder.")


if __name__ == "__main__":
    main()
