# FIFA World Cup Predictor

A machine learning-based FIFA World Cup prediction system that estimates tournament winners using historical international football match data, FIFA rankings, and Monte Carlo simulation.

## Project Overview

This project predicts FIFA World Cup outcomes by combining:

* Historical international football match results
* FIFA team rankings
* Team form analysis
* Random Forest classification
* Monte Carlo tournament simulation

The model is trained on historical match results and then used to simulate tournament outcomes to estimate championship probabilities.

---

## Features

### Match Outcome Prediction

Predicts the probability of:

* Team A win
* Draw
* Team B win

using historical performance indicators.

### Team Form Analysis

Calculates team strength using:

* Recent form points
* Goal difference trends
* Relative performance metrics

### Tournament Simulation

Runs repeated tournament simulations to estimate:

* Winning probabilities
* Most likely champions
* Comparative team performance

---

## Project Structure

```text
fifa-world-cup-predictor/
│
├── data/
│   ├── raw/
│   │   └── results.csv
│   └── teams.csv
│
├── models/
│   ├── match_outcome_model.joblib
│   └── feature_columns.joblib
│
├── src/
│   ├── config.py
│   ├── data_preparation.py
│   ├── train_model.py
│   └── simulate_world_cup.py
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Monte Carlo Simulation

---

## Model Training

Train the model using historical international match data:

```bash
python src/train_model.py
```

The training process:

1. Loads historical match data
2. Extracts team performance features
3. Trains a Random Forest classifier
4. Saves the trained model

---

## Running Tournament Simulations

Run World Cup simulations:

```bash
python src/simulate_world_cup.py --sims 1000 --top-n 32
```

Parameters:

* `--sims`: Number of simulations
* `--top-n`: Number of highest-ranked teams included

---

## Example Output

```text
World Cup Winner Probabilities

France: 14.00%
Argentina: 14.00%
England: 10.00%
Belgium: 8.00%
Netherlands: 8.00%
```

---

## Methodology

### Feature Engineering

The model uses:

* Neutral venue indicator
* Team form points
* Goal difference averages
* Relative strength differences

### Machine Learning Model

Random Forest Classifier

Used for:

* Robust classification
* Non-linear pattern detection
* Stable performance on historical football data

### Tournament Simulation

Monte Carlo simulation repeatedly generates tournament brackets and predicts outcomes to estimate winner probabilities.

---

## Future Improvements

* Full FIFA World Cup 2026 group-stage format simulation
* Live football data API integration
* Elo rating integration
* Interactive dashboard visualization

---

