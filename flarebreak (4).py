"""
=====================================================================
FLAREBREAK
An AI Early-Warning System for Predicting Dangerous Solar Radiation
Storms, Using Historical Solar Flare Data + a Live Solar Flare Monitor
=====================================================================

HOW TO RUN THIS (Colab / Kaggle / Jupyter):
  1. Paste this ENTIRE file into ONE notebook cell.
  2. Run that cell (Shift+Enter). It will:
       - train the historical model and print results
       - run a live custom-region test
       - open a LIVE graph of real-time solar flare activity that
         refreshes automatically every 60 seconds by itself
  3. In a NEW cell, predict a brand new region like this:

        predict_new_region({
            "activity": 1,
            "evolution": 3,
            "prev_24hr_activity": 3,
            "area": 2,
        })

     That's it - one function, one dictionary, no setup needed.

HOW TO RUN THIS (plain Python / terminal):
  pip install ucimlrepo scikit-learn pandas matplotlib requests imbalanced-learn
  python flarebreak.py

DATASETS:
  1. UCI Machine Learning Repository - Solar Flare Dataset (historical)
     https://archive.ics.uci.edu/dataset/89/solar+flare
  2. NOAA SWPC - Live GOES X-ray Flux (real-time, updates every ~1 min)
     https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import requests
from datetime import datetime, timezone

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
# PART B: LIVE SOLAR FLARE MONITOR (real-time GOES X-ray flux)
# =====================================================================
#
# WHY THIS IS SEPARATE FROM PART A:
#   Part A (UCI) is a table of individual active regions from 1988 -
#   no timestamps, no shared ID with live data.
#   Part B is a continuous, real-time stream of the Sun's actual
#   X-ray brightness, measured by NOAA's GOES satellites every minute.
#   This is the exact measurement NOAA uses to officially classify
#   solar flares as A/B/C/M/X class, right now, as they happen.
#   They can't be honestly merged row-by-row, so FlareBreak treats
#   them as two complementary layers:
#     Part A -> "Is this TYPE of region historically risky?"
#     Part B -> "What is the Sun actually doing RIGHT NOW?"
#
NOAA_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
LIVE_REFRESH_SECONDS = 60      # NOAA itself only updates this feed about once a minute
ANIMATION_INTERVAL_SECONDS = 1  # but we redraw once a second so the graph visibly moves
FETCH_EVERY_N_FRAMES = LIVE_REFRESH_SECONDS // ANIMATION_INTERVAL_SECONDS

# Cache so we only hit NOAA once a minute, even though we redraw every second
_LIVE_STATE = {"data": None, "using_real_data": True}

# NOAA's official flare classification bands: (lower bound, letter, base for magnitude)
FLARE_BANDS = [
    (1e-4, "X", 1e-4),
    (1e-5, "M", 1e-5),
    (1e-6, "C", 1e-6),
    (1e-7, "B", 1e-7),
    (0.0,  "A", 1e-8),
]
BAND_COLORS = {"A": "#E8F4FD", "B": "#CFE8FB", "C": "#FDF3D0", "M": "#FBDCC0", "X": "#F8B8B0"}
BAND_EDGES = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 10**-3.3]


def classify_flare(flux):
    """Converts a raw X-ray flux value (W/m^2) into NOAA's official
    A/B/C/M/X flare class notation (e.g. 'M2.3')."""
    if flux <= 0:
        return "Below A (background)"
    for lower_bound, letter, base in FLARE_BANDS:
        if flux >= lower_bound:
            return f"{letter}{flux / base:.1f}"
    return "Below A (background)"


def fetch_live_xray_data(url=NOAA_XRAY_URL, timeout=15):
    """
    Downloads the real, live rolling 6-hour GOES X-ray flux feed and
    returns the long-wavelength channel (0.1-0.8nm), which is the
    channel NOAA's official flare classification is based on.
    Raises an exception if the data can't be reached (e.g. no internet)
    so the caller can decide how to handle it.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    if df["time_tag"].dt.tz is not None:
        df["time_tag"] = df["time_tag"].dt.tz_localize(None)
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")

    long_channel = df[df["energy"].astype(str) == "0.1-0.8nm"].copy()
    long_channel = long_channel.sort_values("time_tag").reset_index(drop=True)
    long_channel = long_channel.dropna(subset=["flux"])

    if long_channel.empty:
        raise ValueError("No 0.1-0.8nm records found in the live feed.")
    return long_channel


def make_synthetic_xray_data(n=360):
    """
    Synthetic fallback data (same shape as the real feed) used only if
    there is no internet connection, purely so the graph can still be
    demonstrated. Clearly flagged as synthetic in the plot title.
    """
    times = pd.date_range(end=datetime.now(timezone.utc).replace(tzinfo=None), periods=n, freq="1min")
    baseline = np.random.uniform(1e-7, 3e-7, n)
    spike_start = int(n * 0.8)
    spike = np.zeros(n)
    spike[spike_start:spike_start + 15] = np.geomspace(1e-7, 3e-5, 15)
    return pd.DataFrame({"time_tag": times, "flux": baseline + spike})


def draw_flare_bands(ax, x_start):
    """Draws shaded horizontal bands for each flare class (A/B/C/M/X)."""
    labels = ["A", "B", "C", "M", "X"]
    for i, label in enumerate(labels):
        ax.axhspan(BAND_EDGES[i], BAND_EDGES[i + 1], color=BAND_COLORS[label],
                   alpha=0.5, zorder=0)
        ax.text(x_start, np.sqrt(BAND_EDGES[i] * BAND_EDGES[i + 1]), f" {label}",
                fontsize=11, fontweight="bold", color="#555555",
                va="center", ha="left")


def update_live_flare_plot(frame, ax, fig):
    """
    Called automatically every ANIMATION_INTERVAL_SECONDS (1 second) by
    the animation below. To be honest to the real data, it only
    fetches a fresh reading from NOAA once every FETCH_EVERY_N_FRAMES
    (60 seconds) - but it redraws every second so the graph visibly
    moves in between: a pulsing marker at the latest reading, a
    real-time-moving "Now" line, and a blinking LIVE indicator.
    """
    # ---- Fetch fresh data only once a minute; reuse the cached copy the rest of the time ----
    need_fetch = (_LIVE_STATE["data"] is None) or (frame % FETCH_EVERY_N_FRAMES == 0)
    if need_fetch:
        try:
            _LIVE_STATE["data"] = fetch_live_xray_data()
            _LIVE_STATE["using_real_data"] = True
        except Exception as e:
            _LIVE_STATE["using_real_data"] = False
            if _LIVE_STATE["data"] is None:
                _LIVE_STATE["data"] = make_synthetic_xray_data()
            print(f"[WARNING] Could not fetch live NOAA data ({e}). "
                  f"Showing synthetic placeholder data instead.")

    data = _LIVE_STATE["data"]
    using_real_data = _LIVE_STATE["using_real_data"]

    ax.clear()
    draw_flare_bands(ax, data["time_tag"].iloc[0])
    ax.plot(data["time_tag"], data["flux"], color="#0B3D91", linewidth=1.8,
            label="GOES X-ray Flux (1-8 \u00c5)", zorder=5)

    # ---- Elements that move every second, even between data refreshes ----
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    ax.axvline(now, color="#D62828", linestyle=":", linewidth=1.5, alpha=0.85,
               zorder=6, label="Now")

    # Pulsing marker on the most recent reading (breathing size/opacity)
    pulse = abs(np.sin(frame * 0.35))
    latest_x = data["time_tag"].iloc[-1]
    latest_y = data["flux"].iloc[-1]
    ax.scatter([latest_x], [latest_y], s=90 + 70 * pulse, color="#D62828",
               alpha=0.35 + 0.5 * pulse, zorder=7, edgecolors="none")
    ax.scatter([latest_x], [latest_y], s=40, color="#D62828", zorder=8)

    # Blinking "LIVE" indicator (toggles roughly twice a second)
    blink_on = (frame % 2 == 0)
    live_color = "#D62828" if blink_on else "#F4A6A6"
    ax.text(0.99, 1.03, "\u25CF LIVE", transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=live_color, ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_ylim(1e-9, 10**-3.3)
    x_max = max(data["time_tag"].iloc[-1], pd.Timestamp(now))
    ax.set_xlim(data["time_tag"].iloc[0], x_max + pd.Timedelta(minutes=2))
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("X-ray Flux (W/m\u00b2)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, which="major", axis="x", alpha=0.3)

    flare_class = classify_flare(latest_y)
    checked_at = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    title_prefix = "" if using_real_data else "[SYNTHETIC DEMO DATA - NO INTERNET] "
    ax.set_title(
        f"{title_prefix}FlareBreak Live Solar Flare Monitor\n"
        f"Current class: {flare_class}   |   Data time: {data['time_tag'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}"
        f"   |   Checked: {checked_at}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()

    # Only print to the console when a real fetch happens, so it doesn't spam every second
    if need_fetch:
        print(f"[{checked_at}] Latest flux: {latest_y:.2e} W/m\u00b2  ->  Class: {flare_class}")


def run_live_flare_monitor():
    """
    Opens a live, continuously updating graph of real-time solar
    flare activity. Redraws every second (so it visibly moves - a
    pulsing marker, a moving "Now" line, a blinking LIVE indicator)
    while only pulling fresh data from NOAA once every
    LIVE_REFRESH_SECONDS, matching how often NOAA itself updates it.
    Runs for as long as the window stays open - no need to re-run
    anything by hand.
    """
    print("\n" + "=" * 60)
    print("PART B: LIVE SOLAR FLARE MONITOR")
    print("=" * 60)
    print(f"Opening live graph - visibly updates every "
          f"{ANIMATION_INTERVAL_SECONDS}s, pulls fresh NOAA data every "
          f"{LIVE_REFRESH_SECONDS}s. Close the window to end.\n")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    try:
        fig.canvas.manager.set_window_title("FlareBreak - Live Solar Flare Monitor")
    except Exception:
        pass

    ani = FuncAnimation(
        fig, update_live_flare_plot, fargs=(ax, fig),
        interval=ANIMATION_INTERVAL_SECONDS * 1000, cache_frame_data=False
    )
    plt.show()
    return ani  # keep a reference so the animation isn't garbage-collected


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

# ---- Launch the live, continuously auto-refreshing solar flare graph ----
# This runs last and stays open, refreshing by itself every 60 seconds -
# perfect to leave running during a live presentation.
_live_animation = run_live_flare_monitor()


