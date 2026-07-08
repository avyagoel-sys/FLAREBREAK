# FlareBreak

AI early-warning system that predicts dangerous solar radiation storms
to protect astronauts, using historical solar flare data (UCI) and
real-time NOAA proton flux data.

## How to run
pip install ucimlrepo scikit-learn pandas matplotlib requests
python flarebreak.py

## What it does
- Trains a model on historical solar active-region data to predict
  dangerous (M/X-class) flares
- Monitors live NOAA proton flux data against official radiation
  storm thresholds
- Outputs an early-warning alert with a risk score
