"""
Training script with Optuna hyperparameter optimization.
Temporal split is preserved, no folds are used.
"""

import optuna
import xgboost as xb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def hyper() -> None:
    # ----------------------------
    # LOAD DATA
    # ----------------------------
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Only training data
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(
            f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train set."
        )

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # ----------------------------
    # TEMPORAL SPLIT
    # ----------------------------
    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Verify temporal correctness
    if val_split[constants.COL_TIMESTAMP].min() <= train_split[constants.COL_TIMESTAMP].max():
        raise ValueError("Temporal split is incorrect.")

    print("✅ Temporal split validation passed.")

    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------
    print("\nComputing aggregate features on train split only...")
    train_split = add_aggregate_features(train_split.copy(), train_split)
    val_split = add_aggregate_features(val_split.copy(), train_split)

    print("Handling missing values...")
    train_split = handle_missing_values(train_split, train_split)
    val_split = handle_missing_values(val_split, train_split)

    # ----------------------------
    # FEATURE LIST
    # ----------------------------
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]

    features = [c for c in train_split.columns if c not in exclude_cols]

    # Drop object cols (safety)
    obj_cols = train_split[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in obj_cols]

    X_train = train_split[features]
    y_train = train_split[config.TARGET]
    X_val = val_split[features]
    y_val = val_split[config.TARGET]

    print(f"Training features: {len(features)}")

    # -----------------------------------
    # OPTUNA OBJECTIVE
    # -----------------------------------
    def objective(trial):

        params = {
            "objective": "rmse",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "seed": config.RANDOM_STATE,

            # --------------------------------------------------
            # Темп обучения — оставляем около твоего 0.01
            # --------------------------------------------------
            "learning_rate": trial.suggest_float("learning_rate", 0.006, 0.02),

            # --------------------------------------------------
            # Ключевой параметр — расширяем сильно
            # 31 было отличным, но можно лучше
            # --------------------------------------------------
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),

            # --------------------------------------------------
            # Глубина — оставляем свободной
            # --------------------------------------------------
            "max_depth": -1,

            # --------------------------------------------------
            # Основные параметры регуляризации
            # --------------------------------------------------
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 0.5),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 0.5),

            # --------------------------------------------------
            # Control leaf-wise overfitting
            # --------------------------------------------------
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 150),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 0.1),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),

            # --------------------------------------------------
            # Кол-во деревьев — много, т.к. early stopping
            # --------------------------------------------------
            "n_estimators": 5000,

            # --------------------------------------------------
            # feature/bagging — оставляем около твоих лучших 0.8
            # --------------------------------------------------
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),

            # --------------------------------------------------
            # Бины — немного влияет на качество
            # --------------------------------------------------
            "max_bin": trial.suggest_int("max_bin", 200, 350),
        }

        model = xb.XGBRegressor(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                xb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)
            ],
        )

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        return rmse

    # ----------------------------
    # RUN OPTUNA
    # ----------------------------
    print("\n🔍 Running Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.N_TRIALS)

    print("\nBest params:")
    print(study.best_params)
    print(f"Best RMSE: {study.best_value:.4f}")

    # ----------------------------
    # TRAIN FINAL MODEL WITH BEST PARAMS
    # ----------------------------
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    })

    print("\nTraining final model with best parameters...")
    model = xb.XGBRegressor(**best_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            xb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)
        ],
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)

    print(f"\nFinal Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # ----------------------------
    # SAVE FINAL MODEL
    # ----------------------------
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / config.MODEL_FILENAME

    model.booster_.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    hyper()


