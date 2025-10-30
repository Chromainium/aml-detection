# aml-detection

aml_detection/
├── config/
│   ├── __init__.py
│   ├── features.py          # FEATURE_CONFIGS dict with named feature sets
│   └── models.py            # TrainingConfig, GNNConfig, MLPConfig, XGBoostConfig
├── data/
│   ├── __init__.py
│   ├── loader.py            # Load either pickled graphs or CSVs
│   ├── preprocessor.py      # Temporal splits, bank filtering
│   └── features.py          # FeatureEngineer class (graph → PyG conversion)
├── models/
│   ├── __init__.py
│   ├── base.py              # BaseModel abstract class
│   ├── gnn.py               # EdgeClassifierGNN
│   ├── mlp.py               # EdgeFeatureClassifier (MLPEdgeClassifier)
│   └── xgboost_wrapper.py   # XGBoostEdgeClassifier
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Trainer class
│   └── evaluator.py         # Evaluator class (metrics, risk scoring)
├── scenarios/
│   ├── __init__.py
│   ├── scenario1.py         # Perspective of single bank
│   └── scenario2.py         # Perspective of single cluster
├── examples/
│   ├── compare_models.py    # Compare GNN vs MLP vs XGBoost
│   └── feature_ablation.py  # Test different feature sets
├── checkpoints/             # Stores best_model.pt
├── main.py                  # Entry point
└── README.md
