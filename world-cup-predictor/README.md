# FIFA World Cup Predictor

A machine learning-based football tournament prediction system that estimates championship probabilities using historical international match results, FIFA rankings, and Monte Carlo simulation.

## Overview

This project predicts football tournament outcomes by combining:

* Historical international football match data
* FIFA team rankings
* Team form analysis
* Random Forest classification
* Monte Carlo tournament simulation

The model simulates knockout-style tournament scenarios to estimate winner probabilities.

---

## Features

* Match outcome prediction
* Team strength evaluation
* Tournament simulation
* Winner probability estimation

---

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Joblib

---

## Project Structure

```text
data/        # Historical match and team data
models/      # Trained ML model
src/         # Source code
README.md
requirements.txt
```

---

## How to Run

Train the model:

```bash
python src/train_model.py
```

Run tournament simulation:

```bash
python src/simulate_world_cup.py --sims 1000 --top-n 32
```

---

## Example Output

```text
France: 14.00%
Argentina: 14.00%
England: 10.00%
```

---

## Future Improvements

* Official FIFA group-stage simulation
* Live football data API integration
* Elo rating integration
* Interactive dashboard

---


Sabancı University
Computer Science & Management Double Major
