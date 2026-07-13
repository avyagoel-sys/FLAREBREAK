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

# Imports a tool that lets us control which warnings Python shows us
import warnings
# Tells Python to hide harmless warning messages so the output stays clean
warnings.filterwarnings("ignore")

# pandas: lets us load and work with data tables (like Excel sheets, in code)
import pandas as pd
# numpy: lets us do fast math on lists of numbers (arrays)
import numpy as np
# matplotlib.pyplot: the main tool we use to draw charts and graphs
import matplotlib.pyplot as plt
# matplotlib.dates: helper tool for formatting dates/times on chart axes
import matplotlib.dates as mdates
# FuncAnimation: the tool that lets a chart redraw itself automatically over time
from matplotlib.animation import FuncAnimation
# requests: lets Python download data from a website/API, like a browser would
import requests
# datetime/timezone: let us get and work with the current real-world date and time
from datetime import datetime, timezone

# train_test_split: splits our data into a training portion and a testing portion
from sklearn.model_selection import train_test_split
# RandomForestClassifier: the actual AI model we train (a "forest" of decision trees)
from sklearn.ensemble import RandomForestClassifier
# Import several tools used to measure how good our model's predictions are
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve,
    precision_recall_fscore_support
)
# SMOTE: a tool that creates extra synthetic examples of our rare "Dangerous" class
from imblearn.over_sampling import SMOTE

# A fixed number so that "random" choices are the same every time we run the code
RANDOM_SEED = 42
# Actually applies that fixed number to numpy's random number generator
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------
# THESE GLOBALS ARE FILLED IN WHEN THE SCRIPT RUNS. AFTER RUNNING, YOU
# CAN USE MODEL / FEATURE_COLUMNS / X / Y DIRECTLY IN YOUR NOTEBOOK, OR
# JUST CALL predict_new_region({...}) - SEE BOTTOM OF THIS FILE.
# ---------------------------------------------------------------------
# A placeholder for the trained AI model - starts empty, filled in later
MODEL = None
# A placeholder for the list of column names the model expects as input
FEATURE_COLUMNS = None
# A placeholder for our input data table (the features/columns) - filled in later
X = None
# A placeholder for our answer key (Safe vs Dangerous labels) - filled in later
Y = None
# The probability cutoff used to decide "Dangerous" vs "Safe" - auto-tuned later
DECISION_THRESHOLD = 0.5  # will be tuned automatically during training


# =====================================================================
# PART A: HISTORICAL SOLAR FLARE MODEL (UCI dataset)
# =====================================================================
# Defines a function that loads the historical solar flare dataset
def load_solar_flare_data():
    """
    Tries to download the REAL UCI Solar Flare dataset.
    Falls back to synthetic placeholder data (same structure) only if
    there is no internet connection available.
    """
    # The names we'll give each column, in the order the real dataset uses
    column_names = [
        "zurich_class", "largest_spot_size", "spot_distribution",
        "activity", "evolution", "prev_24hr_activity",
        "historically_complex", "became_historically_complex",
        "area", "area_largest_spot",
        "c_class_count", "m_class_count", "x_class_count"
    ]

    # ---- Attempt 1: official ucimlrepo package ----
    # Try this block of code; if anything fails, jump to "except" instead of crashing
    try:
        # Imports the official UCI dataset-downloading tool
        from ucimlrepo import fetch_ucirepo
        # Downloads dataset #89 (the Solar Flare dataset) from UCI's servers
        dataset = fetch_ucirepo(id=89)
        # Joins the input columns and the answer columns into one single table
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        # Renames the columns to our own clear names from the list above
        df.columns = column_names
        # Prints a success message showing how many rows we loaded
        print(f"[OK] Loaded REAL UCI Solar Flare dataset "
              f"({len(df)} active-region records).")
        # Sends back the data table, plus True meaning "this is real data"
        return df, True
    # If downloading failed for any reason, catch the error quietly
    except Exception:
        # Do nothing here - just move on to try the next method below
        pass

    # ---- Attempt 2: raw CSV mirror on GitHub (same dataset, re-hosted) ----
    # Try this backup method if the first attempt didn't work
    try:
        # The web address of a backup copy of the same dataset on GitHub
        url = ("https://raw.githubusercontent.com/"
               "Ahmad-Alaziz/Solar-Flare-Detection-AI/main/data/flare.data2")
        # Reads that file directly into a data table, using space as the separator
        df = pd.read_csv(url, sep=r"\s+", skiprows=1, header=None,
                          names=column_names)
        # Prints a success message for this backup method
        print(f"[OK] Loaded REAL UCI Solar Flare dataset via mirror "
              f"({len(df)} records).")
        # Sends back the data table, plus True meaning "this is real data"
        return df, True
    # If this backup also failed, catch the error quietly
    except Exception:
        # Do nothing here - just fall through to the synthetic fallback below
        pass

    # ---- Fallback: synthetic data (SAME SCHEMA, clearly labeled) ----
    # Warns the person clearly that we couldn't get real data this time
    print("[WARNING] No internet connection detected. Using SYNTHETIC "
          "placeholder data with the same structure as the real UCI "
          "dataset, purely so the pipeline can be demonstrated.\n"
          "          -> Re-run with an internet connection for real data.")
    # How many fake rows of data to generate
    n = 1000
    # Builds a fake data table with random values, matching the real dataset's shape
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
    # Computes a simple "riskiness score" for each fake row, based on a few columns
    risk_score = (df["activity"] == 1).astype(int) + (df["evolution"] == 3).astype(int) \
                 + (df["prev_24hr_activity"] == 3).astype(int)
    # Generates a fake count of C-class flares, biased higher when risk_score is higher
    df["c_class_count"] = np.random.poisson(1 + risk_score)
    # Generates a fake count of M-class flares, similarly biased by risk
    df["m_class_count"] = np.random.poisson(0.3 * risk_score)
    # Generates a fake count of X-class flares, similarly biased by risk
    df["x_class_count"] = np.random.poisson(0.05 * risk_score)
    # Sends back the fake data table, plus False meaning "this is NOT real data"
    return df, False


# Defines a function that cleans and prepares the data for the AI model
def preprocess(df):
    """
    Encodes categorical columns and engineers the target label:
    DANGEROUS (1) if the region produced an M-class or X-class flare,
    SAFE (0) otherwise.
    """
    # Makes a safe copy of the data so we don't accidentally change the original
    df = df.copy()
    # The columns that contain letters/categories instead of plain numbers
    categorical_cols = ["zurich_class", "largest_spot_size", "spot_distribution"]
    # Converts each category column into several 0/1 columns (one per category)
    df = pd.get_dummies(df, columns=categorical_cols)

    # Creates our target label: 1 if there was any M-class or X-class flare, else 0
    df["dangerous_flare"] = (
        (df["m_class_count"] > 0) | (df["x_class_count"] > 0)
    ).astype(int)

    # Builds the list of input columns, excluding the raw flare counts and the label itself
    feature_cols = [c for c in df.columns if c not in
                    ["c_class_count", "m_class_count", "x_class_count", "dangerous_flare"]]

    # Returns the input columns (X) and the target label column (y) separately
    return df[feature_cols], df["dangerous_flare"]


# Defines a function that actually trains the AI model
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
    # Lets this function update the DECISION_THRESHOLD variable defined above
    global DECISION_THRESHOLD

    # Carve a validation set out of the TRAINING data only
    # Splits the training data again: 80% to actually train on, 20% to tune with
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    # Oversample the rare "Dangerous" class in the training portion only
    # Counts how many "Dangerous" examples exist in the training portion
    n_minority = y_tr.sum()
    # Picks a safe number of "neighbor" points for SMOTE to use (never more than available)
    k_neighbors = min(5, max(1, n_minority - 1))
    # Sets up the SMOTE oversampling tool with that neighbor count
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
    # Actually generates the extra synthetic "Dangerous" examples
    X_resampled, y_resampled = smote.fit_resample(X_tr, y_tr)

    # Creates the Random Forest model: 300 decision trees, each limited in depth
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_SEED
    )
    # Actually trains the model on the SMOTE-balanced training data
    model.fit(X_resampled, y_resampled)

    # Tune the decision threshold on the VALIDATION set (not test set)
    # Asks the model for its predicted "Dangerous" probability on the validation set
    val_proba = model.predict_proba(X_val)[:, 1]
    # Calculates precision and recall at many different possible cutoff thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    # Calculates the F1 score (a precision/recall balance) at each threshold
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    # If we actually got threshold options to choose from...
    if len(thresholds) > 0:
        # Finds which threshold gave the single best F1 score
        best_idx = np.argmax(f1_scores[:-1])
        # Saves that best threshold as our new official decision cutoff
        DECISION_THRESHOLD = float(thresholds[best_idx])
    # Otherwise (edge case with no valid thresholds)...
    else:
        # Just fall back to the standard 50% cutoff
        DECISION_THRESHOLD = 0.5

    # Prints out what threshold got chosen, for transparency
    print(f"[Tuning] Best decision threshold found: {DECISION_THRESHOLD:.3f} "
          f"(instead of the default 0.5)")

    # Sends back the fully trained model
    return model


# Defines a function that measures and reports how good the model actually is
def evaluate_model(model, X_test, y_test):
    # Asks the model for its predicted "Dangerous" probability on the untouched test set
    y_proba = model.predict_proba(X_test)[:, 1]
    # Converts those probabilities into final Yes/No predictions using our tuned threshold
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)

    # Prints a section header for readability
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    # Shows exactly which threshold was used for these results
    print(f"Decision threshold used: {DECISION_THRESHOLD:.3f}")
    # Shows the overall percentage of correct predictions
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Classification Report:")
    # Prints a detailed breakdown of precision/recall/F1 for both Safe and Dangerous
    print(classification_report(y_test, y_pred, target_names=["Safe", "Dangerous"], zero_division=0))

    # Calculates precision, recall, and F1 specifically for the Dangerous class
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    # Prints recall clearly, with an explanation of why it matters most
    print(f"--> Dangerous-class Recall: {rec:.3f}  "
          f"(this is the number that matters most for an early-warning "
          f"system - it's the fraction of REAL dangerous flares the "
          f"model actually caught)")
    # Prints precision clearly, with a plain-English explanation
    print(f"--> Dangerous-class Precision: {prec:.3f}  "
          f"(of everything flagged as dangerous, how many actually were)")

    # Only calculate ROC-AUC if the test set actually contains both classes
    if len(set(y_test)) > 1:
        # Prints the ROC-AUC score, another overall quality measure
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.3f}")

    # Builds the confusion matrix: counts of correct/incorrect predictions per class
    cm = confusion_matrix(y_test, y_pred)
    # Prepares a visual display object for that confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Dangerous"])
    # Creates a new blank chart canvas to draw on
    fig, ax = plt.subplots(figsize=(5, 5))
    # Draws the confusion matrix onto that canvas, colored in shades of blue
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    # Adds a title to the chart
    ax.set_title("FlareBreak: Confusion Matrix")
    # Automatically adjusts spacing so nothing overlaps
    plt.tight_layout()
    # Saves the chart as an image file
    plt.savefig("flarebreak_confusion_matrix.png", dpi=150)
    # Displays the chart in a window (pauses here until you close it)
    plt.show()
    # Closes the chart to free up memory
    plt.close()
    # Confirms the file was saved
    print("[Saved] flarebreak_confusion_matrix.png")


# Defines a function that shows which input features matter most to the model
def plot_feature_importance(model, feature_names):
    # Pulls the model's built-in "importance score" for each feature
    importances = pd.Series(model.feature_importances_, index=feature_names)
    # Sorts them and keeps only the top 10 most important ones
    importances = importances.sort_values(ascending=True).tail(10)

    # Creates a new blank chart canvas, sized for a horizontal bar chart
    plt.figure(figsize=(8, 6))
    # Draws the importances as horizontal bars
    importances.plot(kind="barh", color="#0B3D91")
    # Adds a title
    plt.title("FlareBreak: Top 10 Most Important Predictive Features")
    # Labels the x-axis
    plt.xlabel("Importance")
    # Adjusts spacing automatically
    plt.tight_layout()
    # Saves the chart as an image file
    plt.savefig("flarebreak_feature_importance.png", dpi=150)
    # Displays the chart (pauses until closed)
    plt.show()
    # Frees up memory
    plt.close()
    # Confirms the save
    print("[Saved] flarebreak_feature_importance.png")


# Defines the function YOU use to test the model on a brand new made-up region
def predict_new_region(region_features: dict, threshold=None):
    """
    *** THIS IS THE FUNCTION YOU USE TO TEST NEW DATA ***

    Give it a dictionary describing a solar active region, and it
    tells you the model's predicted risk of a dangerous flare.
    Uses the automatically-tuned DECISION_THRESHOLD by default (found
    during training to best balance catching real danger vs. false
    alarms) unless you explicitly pass a different one.
    """
    # If no custom threshold was given, use the one found during training
    if threshold is None:
        threshold = DECISION_THRESHOLD

    # Safety check: make sure the model has actually been trained already
    if MODEL is None or FEATURE_COLUMNS is None:
        # Warn the person clearly instead of crashing with a confusing error
        print("[ERROR] The model hasn't been trained yet. Run this script "
              "(or this notebook cell) fully first, then try again.")
        # Stop here and return "nothing" for both values
        return None, None

    # Builds one data row from the dictionary given, filling missing fields with 0
    row = pd.DataFrame([{col: region_features.get(col, 0) for col in FEATURE_COLUMNS}])
    # Asks the model for its predicted "Dangerous" probability for this one row
    risk = MODEL.predict_proba(row)[0, 1]

    # If the risk is at or above our threshold...
    if risk >= threshold:
        # Build an alert message recommending caution
        alert = (f"[ALERT] Elevated radiation storm risk detected "
                  f"({risk*100:.1f}% probability). Recommend astronauts "
                  f"shelter or delay scheduled spacewalks.")
    # Otherwise (risk is below threshold)...
    else:
        # Build a message saying things look safe
        alert = (f"[CLEAR] Low radiation storm risk "
                  f"({risk*100:.1f}% probability). Normal operations "
                  f"can proceed.")

    # Prints the alert message to the screen
    print(alert)
    # Sends back both the raw risk number and the message, in case you want to reuse them
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
# The web address of NOAA's real, live, constantly-updating X-ray data feed
NOAA_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
# How often (in seconds) we actually ask NOAA for a fresh reading
LIVE_REFRESH_SECONDS = 60      # NOAA itself only updates this feed about once a minute
# How often (in seconds) we redraw the chart on screen, for smooth visible motion
ANIMATION_INTERVAL_SECONDS = 1  # but we redraw once a second so the graph visibly moves
# Calculates how many 1-second redraws happen between each real 60-second data fetch
FETCH_EVERY_N_FRAMES = LIVE_REFRESH_SECONDS // ANIMATION_INTERVAL_SECONDS

# Cache so we only hit NOAA once a minute, even though we redraw every second
# A shared "memory box" that remembers the last data we downloaded, between redraws
_LIVE_STATE = {"data": None, "using_real_data": True}

# NOAA's official flare classification bands: (lower bound, letter, base for magnitude)
# The official boundaries and labels NOAA uses to classify flare strength
FLARE_BANDS = [
    (1e-4, "X", 1e-4),
    (1e-5, "M", 1e-5),
    (1e-6, "C", 1e-6),
    (1e-7, "B", 1e-7),
    (0.0,  "A", 1e-8),
]
# The background shading color used for each flare class on the chart
BAND_COLORS = {"A": "#E8F4FD", "B": "#CFE8FB", "C": "#FDF3D0", "M": "#FBDCC0", "X": "#F8B8B0"}
# The exact top/bottom flux values where each colored band starts and ends
BAND_EDGES = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 10**-3.3]


# Defines a function that converts a raw flux number into a flare class label
def classify_flare(flux):
    """Converts a raw X-ray flux value (W/m^2) into NOAA's official
    A/B/C/M/X flare class notation (e.g. 'M2.3')."""
    # If the flux is zero or negative (shouldn't normally happen), treat as background
    if flux <= 0:
        return "Below A (background)"
    # Check each flare band from strongest (X) to weakest (A)
    for lower_bound, letter, base in FLARE_BANDS:
        # If this flux value is at least as big as this band's lower edge...
        if flux >= lower_bound:
            # Return the class letter plus a decimal magnitude (like "M2.3")
            return f"{letter}{flux / base:.1f}"
    # Fallback in case nothing matched (shouldn't normally happen)
    return "Below A (background)"


# Defines a function that downloads the real, live NOAA X-ray flux data
def fetch_live_xray_data(url=NOAA_XRAY_URL, timeout=15):
    """
    Downloads the real, live rolling 6-hour GOES X-ray flux feed and
    returns the long-wavelength channel (0.1-0.8nm), which is the
    channel NOAA's official flare classification is based on.
    Raises an exception if the data can't be reached (e.g. no internet)
    so the caller can decide how to handle it.
    """
    # Sends a web request to NOAA's server asking for the live data file
    response = requests.get(url, timeout=timeout)
    # If the request failed (e.g. server error), this raises an error immediately
    response.raise_for_status()
    # Converts the downloaded JSON data into a pandas data table
    df = pd.DataFrame(response.json())
    # Converts the plain-text timestamps into real, usable datetime values
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    # If those timestamps came with timezone info attached...
    if df["time_tag"].dt.tz is not None:
        # Strip the timezone info so it can be safely compared later on
        df["time_tag"] = df["time_tag"].dt.tz_localize(None)
    # Makes sure the flux column is treated as actual numbers (not text)
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")

    # Keeps only the rows for the "long" wavelength channel (the one that defines flare class)
    long_channel = df[df["energy"].astype(str) == "0.1-0.8nm"].copy()
    # Sorts the rows in time order, oldest to newest
    long_channel = long_channel.sort_values("time_tag").reset_index(drop=True)
    # Removes any rows where the flux value came back missing/invalid
    long_channel = long_channel.dropna(subset=["flux"])

    # If somehow we ended up with zero usable rows...
    if long_channel.empty:
        # Raise an error so the caller knows this attempt failed
        raise ValueError("No 0.1-0.8nm records found in the live feed.")
    # Sends back the cleaned-up live data table
    return long_channel


# Defines a function that creates fake data to use only if there's no internet
def make_synthetic_xray_data(n=360):
    """
    Synthetic fallback data (same shape as the real feed) used only if
    there is no internet connection, purely so the graph can still be
    demonstrated. Clearly flagged as synthetic in the plot title.
    """
    # Creates a list of fake timestamps, one per minute, ending at the current time
    times = pd.date_range(end=datetime.now(timezone.utc).replace(tzinfo=None), periods=n, freq="1min")
    # Generates a quiet "background" flux level with a little random noise
    baseline = np.random.uniform(1e-7, 3e-7, n)
    # Decides where in the timeline a fake flare "spike" will begin
    spike_start = int(n * 0.8)
    # Starts with an all-zero spike array (no extra flux added anywhere yet)
    spike = np.zeros(n)
    # Fills in a rising-then-present spike shape to simulate a flare event
    spike[spike_start:spike_start + 15] = np.geomspace(1e-7, 3e-5, 15)
    # Combines the background and the spike into one final fake dataset
    return pd.DataFrame({"time_tag": times, "flux": baseline + spike})


# Defines a function that draws the colored A/B/C/M/X background bands on the chart
def draw_flare_bands(ax, x_start):
    """Draws shaded horizontal bands for each flare class (A/B/C/M/X)."""
    # The order of class labels from weakest to strongest
    labels = ["A", "B", "C", "M", "X"]
    # Goes through each label one at a time
    for i, label in enumerate(labels):
        # Draws a shaded horizontal strip on the chart for this flare class's range
        ax.axhspan(BAND_EDGES[i], BAND_EDGES[i + 1], color=BAND_COLORS[label],
                   alpha=0.5, zorder=0)
        # Writes the class letter as a small label inside that strip
        ax.text(x_start, np.sqrt(BAND_EDGES[i] * BAND_EDGES[i + 1]), f" {label}",
                fontsize=11, fontweight="bold", color="#555555",
                va="center", ha="left")


# Defines the function that redraws the live graph - called automatically every second
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
    # Decide whether it's time to actually download new data this round
    need_fetch = (_LIVE_STATE["data"] is None) or (frame % FETCH_EVERY_N_FRAMES == 0)
    # If it's time for a fresh download...
    if need_fetch:
        # Try to actually fetch the real live data
        try:
            # Downloads the newest data and stores it in our shared cache
            _LIVE_STATE["data"] = fetch_live_xray_data()
            # Marks that this data is genuinely real/live
            _LIVE_STATE["using_real_data"] = True
        # If the download failed for any reason...
        except Exception as e:
            # Marks that we're no longer showing real live data
            _LIVE_STATE["using_real_data"] = False
            # Only generate fake data if we don't already have some cached from before
            if _LIVE_STATE["data"] is None:
                _LIVE_STATE["data"] = make_synthetic_xray_data()
            # Warns the person that the live fetch failed this round
            print(f"[WARNING] Could not fetch live NOAA data ({e}). "
                  f"Showing synthetic placeholder data instead.")

    # Pulls the current cached data out for use in this redraw
    data = _LIVE_STATE["data"]
    # Pulls the current "is this real data" flag out for use in this redraw
    using_real_data = _LIVE_STATE["using_real_data"]

    # Wipes the chart clean so we can redraw everything fresh this frame
    ax.clear()
    # Draws the colored A/B/C/M/X background bands first (so the line draws on top)
    draw_flare_bands(ax, data["time_tag"].iloc[0])
    # Draws the actual X-ray flux line using the cached data
    ax.plot(data["time_tag"], data["flux"], color="#0B3D91", linewidth=1.8,
            label="GOES X-ray Flux (1-8 \u00c5)", zorder=5)

    # ---- Elements that move every second, even between data refreshes ----
    # Gets the current real-world time (used to draw the moving "Now" line)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    # Draws a vertical dotted red line at the current real-world time
    ax.axvline(now, color="#D62828", linestyle=":", linewidth=1.5, alpha=0.85,
               zorder=6, label="Now")

    # Pulsing marker on the most recent reading (breathing size/opacity)
    # Calculates a smoothly repeating "pulse" value between 0 and 1 based on the frame number
    pulse = abs(np.sin(frame * 0.35))
    # Grabs the timestamp of the most recent data point
    latest_x = data["time_tag"].iloc[-1]
    # Grabs the flux value of the most recent data point
    latest_y = data["flux"].iloc[-1]
    # Draws a soft, glowing circle behind the latest point that grows/shrinks each frame
    ax.scatter([latest_x], [latest_y], s=90 + 70 * pulse, color="#D62828",
               alpha=0.35 + 0.5 * pulse, zorder=7, edgecolors="none")
    # Draws a small solid dot exactly on the latest data point, on top of the glow
    ax.scatter([latest_x], [latest_y], s=40, color="#D62828", zorder=8)

    # Blinking "LIVE" indicator (toggles roughly twice a second)
    # Decides whether the LIVE text should be "on" (bright) or "off" (faded) this frame
    blink_on = (frame % 2 == 0)
    # Picks the bright color if "on", or a faded pink if "off"
    live_color = "#D62828" if blink_on else "#F4A6A6"
    # Draws the "LIVE" indicator text in the corner of the chart
    ax.text(0.99, 1.03, "\u25CF LIVE", transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=live_color, ha="right", va="bottom")

    # Sets the y-axis to a logarithmic scale (needed since flux values span many powers of 10)
    ax.set_yscale("log")
    # Sets the fixed minimum/maximum range shown on the y-axis
    ax.set_ylim(1e-9, 10**-3.3)
    # Figures out whichever is later: the last data point's time, or right now
    x_max = max(data["time_tag"].iloc[-1], pd.Timestamp(now))
    # Sets the x-axis range so the whole timeline plus a little buffer is visible
    ax.set_xlim(data["time_tag"].iloc[0], x_max + pd.Timedelta(minutes=2))
    # Labels the x-axis
    ax.set_xlabel("Time (UTC)")
    # Labels the y-axis
    ax.set_ylabel("X-ray Flux (W/m\u00b2)")
    # Formats the x-axis tick labels to show just hour:minute
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # Adds faint vertical gridlines for readability
    ax.grid(True, which="major", axis="x", alpha=0.3)

    # Converts the latest flux reading into a flare class label (like "B2.4")
    flare_class = classify_flare(latest_y)
    # Records the exact real-world time this check happened, as text
    checked_at = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    # Adds a warning prefix to the title if we're not using real data right now
    title_prefix = "" if using_real_data else "[SYNTHETIC DEMO DATA - NO INTERNET] "
    # Builds and sets the chart's title, showing the current class and timestamps
    ax.set_title(
        f"{title_prefix}FlareBreak Live Solar Flare Monitor\n"
        f"Current class: {flare_class}   |   Data time: {data['time_tag'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}"
        f"   |   Checked: {checked_at}",
        fontsize=13, fontweight="bold"
    )
    # Shows the legend (labels for the line and the "Now" marker) in the corner
    ax.legend(loc="upper left")
    # Automatically adjusts spacing so nothing overlaps or gets cut off
    fig.tight_layout()

    # Only print to the console when a real fetch happens, so it doesn't spam every second
    # If this round actually involved a fresh data download...
    if need_fetch:
        # Prints the latest reading and its flare class to the console
        print(f"[{checked_at}] Latest flux: {latest_y:.2e} W/m\u00b2  ->  Class: {flare_class}")


# Defines the function that sets up and starts the live, self-updating graph
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
    # Prints a section header
    print("\n" + "=" * 60)
    print("PART B: LIVE SOLAR FLARE MONITOR")
    print("=" * 60)
    # Explains to the person what's about to happen and how often it updates
    print(f"Opening live graph - visibly updates every "
          f"{ANIMATION_INTERVAL_SECONDS}s, pulls fresh NOAA data every "
          f"{LIVE_REFRESH_SECONDS}s. Close the window to end.\n")

    # Creates the blank chart window and drawing area we'll keep reusing
    fig, ax = plt.subplots(figsize=(11, 6.5))
    # Try to give the window a nice title bar (not all systems support this)
    try:
        # Sets the window's title bar text
        fig.canvas.manager.set_window_title("FlareBreak - Live Solar Flare Monitor")
    # If that feature isn't available on this system...
    except Exception:
        # Just skip it silently - not essential
        pass

    # Sets up the animation: call update_live_flare_plot automatically, once per second
    ani = FuncAnimation(
        fig, update_live_flare_plot, fargs=(ax, fig),
        interval=ANIMATION_INTERVAL_SECONDS * 1000, cache_frame_data=False
    )
    # Actually opens the window and starts the animation loop (pauses here until closed)
    plt.show()
    # Sends back the animation object so Python doesn't delete it while it's still running
    return ani  # keep a reference so the animation isn't garbage-collected


# =====================================================================
# RUN EVERYTHING (this executes automatically whether you run this as
# a script with `python flarebreak.py`, OR paste it into a Colab/
# Jupyter cell and run the cell directly)
# =====================================================================
# Prints a starting message so you know the script has begun
print("Starting FlareBreak pipeline...\n")

# Loads the historical dataset (real if possible, synthetic if not)
df, using_real_data = load_solar_flare_data()
# Cleans the data and splits it into inputs (X) and the answer key (Y)
X, Y = preprocess(df)
# Prints what percentage of historical regions were actually "Dangerous"
print(f"\nDangerous flare rate in dataset: {Y.mean()*100:.1f}%")

# Splits the data into a training portion (75%) and a completely separate test portion (25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=RANDOM_SEED, stratify=Y
)

# Trains the model on the training data, and saves it into our global MODEL variable
MODEL = train_model(X_train, y_train)
# Saves the list of column names the model expects, for later use in predict_new_region
FEATURE_COLUMNS = list(X.columns)

# Evaluates the trained model honestly on the untouched test data
evaluate_model(MODEL, X_test, y_test)
# Draws and saves the feature importance chart
plot_feature_importance(MODEL, FEATURE_COLUMNS)

# Prints a section header for the demo
print("\n" + "=" * 60)
print("EARLY WARNING SYSTEM DEMO")
print("=" * 60)
# Runs one example prediction using the first row of the test set, just to show it working
predict_new_region(X_test.iloc[0].to_dict())

# If we ended up using fake historical data earlier...
if not using_real_data:
    # Remind the person clearly, one more time, before moving on
    print("\nReminder: this run used SYNTHETIC placeholder data because "
          "no internet connection was available.")

# Lets the person know Part A is finished and ready to use
print("\nDone. MODEL and FEATURE_COLUMNS are ready to use.")

# =====================================================================
# >>> TEST YOUR OWN REGION HERE <<<
# Edit the numbers below to describe any region you want to check,
# then just re-run this ENTIRE cell (Ctrl+Enter / Shift+Enter).
# You do NOT need a separate cell - this runs automatically below.
# =====================================================================
# Prints a section header for your custom test
print("\n" + "=" * 60)
print("CUSTOM REGION TEST")
print("=" * 60)

# A dictionary describing one made-up solar region you want to test - edit these numbers freely
my_region = {
    "activity": 1,                    # 1 = reduced, 2 = unchanged
    "evolution": 3,                   # 1 = decay, 2 = no growth, 3 = growth
    "prev_24hr_activity": 3,          # 1 = nothing, 2 = one M1, 3 = more than one M1
    "area": 2,                        # 1 = small, 2 = large
}

# Runs the prediction on your custom region and prints the result
predict_new_region(my_region)

# ---- Launch the live, continuously auto-refreshing solar flare graph ----
# This runs last and stays open, refreshing by itself every 60 seconds -
# perfect to leave running during a live presentation.
# Starts the live graph window; this line "pauses" the script here until you close the window
_live_animation = run_live_flare_monitor()
