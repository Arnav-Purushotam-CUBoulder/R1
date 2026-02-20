from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_SEEDS = [11, 17, 23, 31, 47]
BENCHMARK_SUITE = "OpenML-CC18"


@dataclass(frozen=True)
class DatasetSpec:
    task_id: int
    dataset_id: int
    name: str
    version: int
    target_name: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_name: str
    preprocess: str
    configs: tuple[dict[str, object], ...]


DATASET_FILTERS = {
    "suite": BENCHMARK_SUITE,
    "instances_min": 500,
    "instances_max": 20_000,
    "predictor_features_min": 4,
    "predictor_features_max": 60,
    "classes_min": 2,
    "classes_max": 10,
    "missing_values": 0,
    "max_symbolic_features": 1,
}


MODELS = [
    ModelSpec(
        name="Logistic Regression",
        estimator_name="logreg",
        preprocess="scaled",
        configs=tuple(
            {"C": c_value, "class_weight": class_weight}
            for c_value in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            for class_weight in [None, "balanced"]
        ),
    ),
    ModelSpec(
        name="Random Forest",
        estimator_name="rf",
        preprocess="tree",
        configs=tuple(
            {
                "n_estimators": 300,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
            for max_depth in [None, 12, 24]
            for min_samples_leaf in [1, 2]
            for max_features in ["sqrt", 0.5]
        ),
    ),
    ModelSpec(
        name="Extra Trees",
        estimator_name="et",
        preprocess="tree",
        configs=tuple(
            {
                "n_estimators": 300,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
            for max_depth in [None, 12, 24]
            for min_samples_leaf in [1, 2]
            for max_features in ["sqrt", 0.5]
        ),
    ),
    ModelSpec(
        name="HistGradientBoosting",
        estimator_name="hgb",
        preprocess="tree",
        configs=tuple(
            {
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "max_leaf_nodes": max_leaf_nodes,
                "min_samples_leaf": 20,
            }
            for learning_rate in [0.03, 0.1]
            for max_depth in [None, 6, 12]
            for max_leaf_nodes in [15, 31]
        ),
    ),
    ModelSpec(
        name="MLP",
        estimator_name="mlp",
        preprocess="scaled",
        configs=tuple(
            {
                "hidden_layer_sizes": hidden_layers,
                "alpha": alpha,
                "learning_rate_init": learning_rate_init,
            }
            for hidden_layers in [(64,), (128,), (128, 64)]
            for alpha in [1e-5, 1e-4]
            for learning_rate_init in [3e-4, 1e-3]
        ),
    ),
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_datasets() -> list[DatasetSpec]:
    suite = openml.study.get_suite(BENCHMARK_SUITE)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="openml.tasks.functions")
        task_meta = openml.tasks.list_tasks(output_format="dataframe")
    dataset_meta = openml.datasets.list_datasets(output_format="dataframe")

    candidates = (
        task_meta[task_meta["tid"].isin(suite.tasks)][["tid", "did", "target_feature"]]
        .merge(dataset_meta, on="did", how="inner")
        .assign(predictor_features=lambda df: df["NumberOfFeatures"] - 1)
    )

    filtered = candidates[
        candidates["status"].eq("active")
        & candidates["NumberOfClasses"].notna()
        & candidates["NumberOfClasses"].between(DATASET_FILTERS["classes_min"], DATASET_FILTERS["classes_max"])
        & candidates["NumberOfInstances"].between(DATASET_FILTERS["instances_min"], DATASET_FILTERS["instances_max"])
        & candidates["predictor_features"].between(
            DATASET_FILTERS["predictor_features_min"],
            DATASET_FILTERS["predictor_features_max"],
        )
        & candidates["NumberOfMissingValues"].fillna(0).eq(DATASET_FILTERS["missing_values"])
        & candidates["NumberOfSymbolicFeatures"].fillna(0).le(DATASET_FILTERS["max_symbolic_features"])
    ].sort_values(["NumberOfInstances", "name", "did"])

    return [
        DatasetSpec(
            task_id=int(row.tid),
            dataset_id=int(row.did),
            name=str(row.name),
            version=int(row.version),
            target_name=str(row.target_feature),
        )
        for row in filtered.itertuples(index=False)
    ]


def load_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, np.ndarray, dict]:
    dataset = openml.datasets.get_dataset(spec.dataset_id, download_data=True)
    X, y, categorical_indicator, feature_names = dataset.get_data(
        target=spec.target_name,
        dataset_format="dataframe",
    )
    if any(categorical_indicator):
        raise ValueError(f"{spec.name} includes non-numeric predictor columns after filtering.")
    X = X[feature_names].astype(float)
    y = LabelEncoder().fit_transform(y.astype(str))
    info = {
        "task_id": spec.task_id,
        "dataset_id": spec.dataset_id,
        "dataset": spec.name,
        "version": spec.version,
        "target_name": spec.target_name,
        "instances": len(X),
        "features": X.shape[1],
        "classes": int(np.unique(y).size),
        "minority_share": float(pd.Series(y).value_counts(normalize=True).min()),
    }
    return X, y.astype(int), info


def make_preprocessor(kind: str, feature_names: list[str]) -> ColumnTransformer:
    if kind == "scaled":
        transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    elif kind == "tree":
        transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    else:
        raise ValueError(f"Unknown preprocessor kind: {kind}")

    return ColumnTransformer(
        transformers=[("num", transformer, feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(model: ModelSpec, params: dict, seed: int):
    if model.estimator_name == "logreg":
        return LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
            **params,
        )
    if model.estimator_name == "rf":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if model.estimator_name == "et":
        return ExtraTreesClassifier(
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if model.estimator_name == "hgb":
        return HistGradientBoostingClassifier(
            random_state=seed,
            early_stopping=False,
            **params,
        )
    if model.estimator_name == "mlp":
        return MLPClassifier(
            random_state=seed,
            activation="relu",
            solver="adam",
            batch_size=128,
            early_stopping=False,
            max_iter=500,
            tol=1e-4,
            **params,
        )
    raise ValueError(f"Unknown estimator {model.estimator_name}")


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    one_hot = np.eye(proba.shape[1])[y_true]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bucket_conf = confidences[mask].mean()
        bucket_acc = correctness[mask].mean()
        ece += mask.mean() * abs(bucket_acc - bucket_conf)
    return float(ece)


def get_probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)
    if scores.ndim == 1:
        probs_1 = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probs_1, probs_1])
    return softmax(scores, axis=1)


def compute_metrics(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X)
    proba = get_probabilities(model, X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "brier": multiclass_brier_score(y, proba),
        "ece": expected_calibration_error(y, proba),
    }


def fit_and_score(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    feature_names: list[str],
    model_spec: ModelSpec,
    params: dict,
    seed: int,
) -> tuple[Pipeline, dict[str, float], float]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(model_spec.preprocess, feature_names)),
            ("estimator", make_estimator(model_spec, params, seed)),
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
    metrics = compute_metrics(pipeline, X_eval, y_eval)
    return pipeline, metrics, fit_time


def split_three_way(X: pd.DataFrame, y: np.ndarray, seed: int):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_benchmark(output_dir: Path) -> None:
    raw_dir = ensure_dir(output_dir / "raw")
    dataset_specs = discover_datasets()
    dataset_rows = []
    run_rows = []
    search_rows = []

    for dataset_spec in dataset_specs:
        print(f"[dataset] {dataset_spec.name} (task {dataset_spec.task_id})", flush=True)
        X, y, dataset_info = load_dataset(dataset_spec)
        dataset_rows.append(dataset_info)
        feature_names = list(X.columns)

        for seed in RANDOM_SEEDS:
            print(f"  [seed] {seed}", flush=True)
            X_train, X_val, X_test, y_train, y_val, y_test = split_three_way(X, y, seed)
            X_train_val = pd.concat([X_train, X_val], axis=0)
            y_train_val = np.concatenate([y_train, y_val])

            for model_spec in MODELS:
                print(f"    [model] {model_spec.name}", flush=True)
                best_record = None
                search_start = time.perf_counter()

                for idx, params in enumerate(model_spec.configs, start=1):
                    record = {
                        "task_id": dataset_spec.task_id,
                        "dataset_id": dataset_spec.dataset_id,
                        "dataset": dataset_spec.name,
                        "seed": seed,
                        "model": model_spec.name,
                        "config_index": idx,
                        "params_json": json.dumps(params, sort_keys=True),
                        "error": "",
                    }
                    try:
                        _, val_metrics, fit_time = fit_and_score(
                            X_train=X_train,
                            y_train=y_train,
                            X_eval=X_val,
                            y_eval=y_val,
                            feature_names=feature_names,
                            model_spec=model_spec,
                            params=params,
                            seed=seed,
                        )
                        record.update(
                            {
                                "val_macro_f1": val_metrics["macro_f1"],
                                "val_accuracy": val_metrics["accuracy"],
                                "val_brier": val_metrics["brier"],
                                "val_ece": val_metrics["ece"],
                                "val_fit_time_sec": fit_time,
                            }
                        )
                        sort_key = (
                            val_metrics["macro_f1"],
                            val_metrics["accuracy"],
                            -val_metrics["brier"],
                            -val_metrics["ece"],
                        )
                        if best_record is None or sort_key > best_record["sort_key"]:
                            best_record = {
                                "params": params,
                                "sort_key": sort_key,
                                "config_index": idx,
                            }
                    except Exception as exc:
                        record.update(
                            {
                                "val_macro_f1": np.nan,
                                "val_accuracy": np.nan,
                                "val_brier": np.nan,
                                "val_ece": np.nan,
                                "val_fit_time_sec": np.nan,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    search_rows.append(record)

                search_time = time.perf_counter() - search_start
                if best_record is None:
                    raise RuntimeError(
                        f"No valid configuration for {dataset_spec.name} / {model_spec.name} / {seed}."
                    )

                final_model, test_metrics, final_fit_time = fit_and_score(
                    X_train=X_train_val,
                    y_train=y_train_val,
                    X_eval=X_test,
                    y_eval=y_test,
                    feature_names=feature_names,
                    model_spec=model_spec,
                    params=best_record["params"],
                    seed=seed,
                )

                predict_start = time.perf_counter()
                _ = final_model.predict(X_test)
                inference_time = time.perf_counter() - predict_start

                run_rows.append(
                    {
                        "task_id": dataset_spec.task_id,
                        "dataset_id": dataset_spec.dataset_id,
                        "dataset": dataset_spec.name,
                        "seed": seed,
                        "model": model_spec.name,
                        "n_train": len(X_train),
                        "n_val": len(X_val),
                        "n_test": len(X_test),
                        "accuracy": test_metrics["accuracy"],
                        "macro_f1": test_metrics["macro_f1"],
                        "brier": test_metrics["brier"],
                        "ece": test_metrics["ece"],
                        "search_time_sec": search_time,
                        "fit_time_sec": final_fit_time,
                        "inference_time_sec": inference_time,
                        "selected_config_index": best_record["config_index"],
                        "selected_params_json": json.dumps(best_record["params"], sort_keys=True),
                    }
                )

    pd.DataFrame(dataset_rows).sort_values(["instances", "dataset"]).to_csv(raw_dir / "dataset_summary.csv", index=False)
    pd.DataFrame(search_rows).to_csv(raw_dir / "search_results.csv", index=False)
    pd.DataFrame(run_rows).to_csv(raw_dir / "run_results.csv", index=False)
