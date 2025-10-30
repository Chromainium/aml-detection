## Project Structure

```
money_laundering_detection/
├── config/
│   ├── features.py          # Feature set configurations
│   └── models.py            # Model hyperparameters
├── data/
│   ├── loader.py            # Data loading utilities
│   ├── preprocessor.py      # Temporal splits and filtering
│   └── features.py          # Feature engineering
├── models/
│   ├── base.py              # Abstract base class
│   ├── gnn.py               # Graph Neural Network
│   ├── mlp.py               # Multi-Layer Perceptrons
│   └── xgboost_wrapper.py   # XGBoost wrapper
├── training/
│   ├── trainer.py           # Training loops
│   └── evaluator.py         # Evaluation and metrics
├── scenarios/
│   ├── scenario1.py         # BigBank perspective
│   └── scenario2.py         # Cluster perspective (future)
└── main.py                  # Entry point
```