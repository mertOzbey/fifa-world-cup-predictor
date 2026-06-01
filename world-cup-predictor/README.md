# FIFA World Cup 2026 Predictor

A machine learning project that predicts FIFA World Cup outcomes using:

- Historical international football match results
- FIFA world rankings
- Random Forest classification
- Monte Carlo tournament simulation

## Features

- Match outcome prediction
- Team strength modeling
- Tournament simulation
- World Cup winner probability estimation

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib

## Run

```bash
python src/train_model.py
python src/simulate_world_cup.py --sims 1000 --top-n 32
```

## Example Output

France: 14%  
Argentina: 14%  
England: 10%

## Future Improvements

- Full 48-team 2026 format
- Group stage simulation
- Elo rating integration
- Live FIFA ranking API integration