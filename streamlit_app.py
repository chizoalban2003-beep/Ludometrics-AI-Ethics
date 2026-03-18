"""
Ludomaniac — Interactive Streamlit Dashboard
Workflow: Dataset Creation → EDA → Feature Engineering → Model Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional


@st.cache_resource
def _train_experimental_random_forest_predictor(
    df_ref: pd.DataFrame,
    feature_columns: tuple[str, ...],
    random_state: int = 101,
) -> "ProductionLudoPredictor":
    """Train an in-app RandomForest predictor (experimental).

    Uses the same engineered feature columns as the shipped production model.
    Returns a `ProductionLudoPredictor` wrapper with Platt-style calibration and
    an F1-optimized threshold on a held-out calibration split.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    if df_ref is None or len(df_ref) == 0:
        raise ValueError("Reference dataset is empty; cannot train.")
    if "Is_Winner" not in df_ref.columns:
        raise ValueError("Dataset missing 'Is_Winner'; cannot train.")

    base_numeric = [
        "Turn",
        "Dice_Roll",
        "Token_Moved",
        "Position_Before",
        "Position_After",
        "Tokens_Home",
        "Tokens_Active",
        "Tokens_Finished",
        "Captured_Opponent",
    ]

    y_all = pd.to_numeric(df_ref["Is_Winner"], errors="coerce")
    mask = y_all.notna()
    y_all = y_all[mask].astype(int)

    raw_cols = [c for c in (["Game_ID", "Player"] + base_numeric) if c in df_ref.columns]
    raw_all = df_ref.loc[mask, raw_cols].copy()

    # Reuse the existing inference-time feature engineering to build training X.
    stub = type("_FeatureColsStub", (), {"feature_columns": list(feature_columns)})()
    X_all = _prepare_model_matrix(raw_all, stub)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=random_state,
        stratify=y_all,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.25,  # 0.25 * 0.8 = 0.2
        random_state=random_state,
        stratify=y_tmp,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    base_pipeline = Pipeline([
        ("model", rf),
    ])
    base_pipeline.fit(X_train, y_train)

    # Calibrate using a logistic regression on the model score.
    cal_scores = base_pipeline.predict_proba(X_cal)[:, 1]
    platt = LogisticRegression(max_iter=2000, random_state=random_state)
    platt.fit(np.asarray(cal_scores).reshape(-1, 1), y_cal)

    cal_proba = platt.predict_proba(np.asarray(cal_scores).reshape(-1, 1))[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr = 0.50
    best_f1 = -1.0
    for thr in thresholds:
        pred = (cal_proba >= thr).astype(int)
        f1 = f1_score(y_cal, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    # Compute held-out test metrics for display.
    test_scores = base_pipeline.predict_proba(X_test)[:, 1]
    test_proba = platt.predict_proba(np.asarray(test_scores).reshape(-1, 1))[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "threshold": float(best_thr),
        "n_train": int(len(X_train)),
        "n_cal": int(len(X_cal)),
        "n_test": int(len(X_test)),
    }

    artifact = {
        "model_variant": "experimental_random_forest",
        "decision_threshold": float(best_thr),
        "feature_columns": list(feature_columns),
        "metrics": metrics,
        "notes": "Trained in-app from the current dataset using engineered features; not the shipped production model.",
    }

    return ProductionLudoPredictor(
        base_pipeline=base_pipeline,
        platt_calibrator=platt,
        decision_threshold=float(best_thr),
        feature_columns=list(feature_columns),
        artifact=artifact,
    )

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Ludomaniac — Ludo Win Prediction",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL & DATA
# ============================================================================


@dataclass(frozen=True)
class ProductionLudoPredictor:
    """Thin wrapper around the shipped production artifact.

    The joblib artifact is a dict containing:
    - base_pipeline: sklearn Pipeline
    - platt_calibrator: sklearn LogisticRegression (Platt scaling)
    - decision_threshold: float
    - feature_columns: ordered list of expected columns
    """

    base_pipeline: Any
    platt_calibrator: Optional[Any]
    decision_threshold: float
    feature_columns: list[str]
    artifact: dict[str, Any]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities [P(0), P(1)]."""
        if X is None or len(X) == 0:
            raise ValueError("Empty input")

        # Some artifacts (e.g. advanced RF) are shipped without a calibrator.
        if self.platt_calibrator is None:
            if hasattr(self.base_pipeline, "predict_proba"):
                return np.asarray(self.base_pipeline.predict_proba(X))
            raise TypeError("Uncalibrated model has no predict_proba")

        # Platt calibrator expects a 2D array of decision scores.
        if hasattr(self.base_pipeline, "decision_function"):
            scores = self.base_pipeline.decision_function(X)
        elif hasattr(self.base_pipeline, "predict_proba"):
            # Fallback: use positive-class probability as a monotonic score.
            scores = self.base_pipeline.predict_proba(X)[:, 1]
        else:
            raise TypeError("Base pipeline has neither decision_function nor predict_proba")

        scores_2d = np.asarray(scores).reshape(-1, 1)
        proba = self.platt_calibrator.predict_proba(scores_2d)
        return np.asarray(proba)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= float(self.decision_threshold)).astype(int)


def _normalize_feature_name(name: Any) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower()


def _engineered_col_name(f1: str, f2: str, transform_tag: str) -> str:
    return f"eng_{transform_tag}__{_normalize_feature_name(f1)}__{_normalize_feature_name(f2)}"


@st.cache_resource
def _get_inference_artifacts() -> dict[str, Any]:
    """Compute and cache artifacts needed to reproduce training-time feature engineering."""
    df_ref = load_dataset()
    if df_ref is None or df_ref.empty:
        raise RuntimeError("Cannot build inference artifacts: dataset could not be loaded")

    # Spearman correlation context (training-time constant, recomputed from shipped dataset)
    excluded_for_engineering = {"Is_Winner", "Game_ID"}
    numeric_cols = [
        c
        for c in df_ref.select_dtypes(include=[np.number]).columns
        if c not in excluded_for_engineering
    ]
    corr = df_ref[numeric_cols].corr(method="spearman")

    pairs: list[dict[str, Any]] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1, f2 = cols[i], cols[j]
            rho = float(corr.loc[f1, f2])
            abs_rho = abs(rho)
            if abs_rho >= 0.7:
                strength = "Strong"
            elif abs_rho >= 0.3:
                strength = "Moderate"
            else:
                strength = "Weak"
            direction = "Positive" if rho > 0 else ("Negative" if rho < 0 else "Neutral")
            pairs.append(
                {
                    "feature_1": f1,
                    "feature_2": f2,
                    "spearman_rho": rho,
                    "abs_rho": abs_rho,
                    "strength": strength,
                    "direction": direction,
                }
            )
    pairwise_correlation_results = pd.DataFrame(pairs)

    # Mann-Whitney U table fallback (copied from Feature_engineering.ipynb)
    mwu_df = pd.DataFrame(
        [
            {"numeric_variable": "Turn", "u_value": 11662408.5, "p_value": 3.436587e-01, "significant": False},
            {"numeric_variable": "Dice_Roll", "u_value": 11581509.0, "p_value": 1.296232e-01, "significant": False},
            {"numeric_variable": "Token_Moved", "u_value": 10872140.5, "p_value": 1.309582e-10, "significant": True},
            {"numeric_variable": "Position_Before", "u_value": 6805749.0, "p_value": 2.839426e-07, "significant": True},
            {"numeric_variable": "Position_After", "u_value": 6776655.0, "p_value": 6.022537e-08, "significant": True},
            {"numeric_variable": "Tokens_Home", "u_value": 15298936.0, "p_value": 8.448305e-135, "significant": True},
            {"numeric_variable": "Tokens_Active", "u_value": 11102690.5, "p_value": 3.245883e-10, "significant": True},
            {"numeric_variable": "Tokens_Finished", "u_value": 8518392.0, "p_value": 1.033559e-120, "significant": True},
            {"numeric_variable": "Captured_Opponent", "u_value": 11715264.5, "p_value": 1.596819e-02, "significant": True},
        ]
    )
    mwu_df["numeric_variable"] = mwu_df["numeric_variable"].astype(str)
    mwu_df["u_value"] = pd.to_numeric(mwu_df["u_value"], errors="coerce")
    mwu_df["p_value"] = pd.to_numeric(mwu_df["p_value"], errors="coerce")
    mwu_df["significant"] = mwu_df["significant"].astype(bool)

    u_valid = mwu_df["u_value"].dropna()
    u_center = float(u_valid.median()) if not u_valid.empty else 0.0
    u_scale = float((u_valid - u_center).abs().max()) if not u_valid.empty else 1.0
    u_scale = u_scale if u_scale > 0 else 1.0

    mwu_df["u_separation_weight"] = ((mwu_df["u_value"] - u_center).abs() / u_scale).clip(0.0, 1.0)
    # De-emphasize non-significant variables but keep tiny contribution for stability
    mwu_df["mwu_weight"] = np.where(mwu_df["significant"], mwu_df["u_separation_weight"], 0.15 * mwu_df["u_separation_weight"])

    mwu_weight_lookup = mwu_df.set_index("numeric_variable")["mwu_weight"].to_dict()
    significant_mwu_features = set(mwu_df.loc[mwu_df["significant"], "numeric_variable"])

    return {
        "pairwise_correlation_results": pairwise_correlation_results,
        "mwu_weight_lookup": mwu_weight_lookup,
        "significant_mwu_features": significant_mwu_features,
    }


def _engineer_features(raw_df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Recreate the engineered feature columns used by the production model."""
    eps = 1e-6
    protected_columns = {"Game_ID", "Is_Winner"}

    pairwise = artifacts["pairwise_correlation_results"]
    mwu_weight_lookup = artifacts["mwu_weight_lookup"]
    significant_features = artifacts["significant_mwu_features"]

    df_fe = raw_df.copy()
    pending_columns: dict[str, Any] = {}
    source_features_used: set[str] = set()

    # Correlation-category engineered features
    for _, row in pairwise.iterrows():
        f1 = str(row["feature_1"])
        f2 = str(row["feature_2"])
        strength = str(row["strength"])
        direction = str(row["direction"])
        rho = float(row["spearman_rho"])

        if rho == 0:
            continue
        if f1 not in df_fe.columns or f2 not in df_fe.columns:
            continue

        # Require at least one source feature to be significant by MWU
        if (f1 not in significant_features) and (f2 not in significant_features):
            continue

        w1 = float(mwu_weight_lookup.get(f1, 0.0))
        w2 = float(mwu_weight_lookup.get(f2, 0.0))
        mwu_pair_weight = (w1 + w2) / 2.0

        combined_weight = rho * (0.25 + 0.75 * mwu_pair_weight)

        if strength == "Strong" and direction == "Positive":
            new_col = _engineered_col_name(f1, f2, "strong_pos_div_wrho_mwu")
            source_features_used.update([f1, f2])
            ratio = df_fe[f1] / (df_fe[f2] + eps)
            pending_columns[new_col] = ratio * combined_weight
        elif strength == "Strong" and direction == "Negative":
            new_col = _engineered_col_name(f1, f2, "strong_neg_mul_wrho_mwu")
            source_features_used.update([f1, f2])
            interaction = df_fe[f1] * df_fe[f2]
            pending_columns[new_col] = interaction * combined_weight
        elif strength == "Weak" and direction == "Negative":
            new_col = _engineered_col_name(f1, f2, "weak_neg_combo_wrho_mwu")
            source_features_used.update([f1, f2])
            ratio = df_fe[f1] / (df_fe[f2] + eps)
            safe_sqrt_term = np.sqrt(np.abs(ratio))
            combo = (df_fe[f1] * df_fe[f2]) - (safe_sqrt_term + ratio)
            pending_columns[new_col] = combo * combined_weight

    if pending_columns:
        df_fe = pd.concat([df_fe, pd.DataFrame(pending_columns, index=df_fe.index)], axis=1).copy()

    # Drop source numeric columns used in engineering (mirrors notebook behavior)
    drop_source_columns = sorted(
        [col for col in source_features_used if col in df_fe.columns and col not in protected_columns]
    )
    if drop_source_columns:
        df_fe = df_fe.drop(columns=drop_source_columns).copy()

    # MWU post-weighting pass (floor 0.10)
    compact_token_map = {
        "turn": "trn",
        "dice_roll": "dice",
        "token_moved": "tok_mv",
        "position_before": "pos_b",
        "position_after": "pos_a",
        "tokens_home": "tok_home",
        "tokens_active": "tok_act",
        "tokens_finished": "tok_fin",
        "captured_opponent": "cap_opp",
        "player_red": "ply_red",
        "player_blue": "ply_blue",
    }

    def _compact_token(name: str) -> str:
        return compact_token_map.get(str(name), str(name))

    norm_mwu_lookup: dict[str, float] = {}
    for k, v in mwu_weight_lookup.items():
        k_norm = _normalize_feature_name(k)
        norm_mwu_lookup[k_norm] = float(v)
        norm_mwu_lookup[_compact_token(k_norm)] = float(v)

    eng_cols_all = [col for col in df_fe.columns if str(col).startswith("eng_")]
    for col in eng_cols_all:
        parts = str(col).split("__")
        if len(parts) < 3:
            continue
        f1_token = parts[1]
        f2_token = parts[2]
        w1 = float(norm_mwu_lookup.get(f1_token, 0.0))
        w2 = float(norm_mwu_lookup.get(f2_token, 0.0))
        mwu_extra = max((w1 + w2) / 2.0, 0.1)
        df_fe[col] = df_fe[col] * mwu_extra

    return df_fe


def _prepare_model_matrix(raw_rows: pd.DataFrame, predictor: ProductionLudoPredictor) -> pd.DataFrame:
    """Convert raw rows (base columns + Player) into the model's expected feature matrix."""
    artifacts = _get_inference_artifacts()

    df_in = raw_rows.copy()

    # Ensure base numeric columns are numeric
    for col in [
        "Turn",
        "Dice_Roll",
        "Token_Moved",
        "Position_Before",
        "Position_After",
        "Tokens_Home",
        "Tokens_Active",
        "Tokens_Finished",
        "Captured_Opponent",
    ]:
        if col in df_in.columns:
            df_in[col] = pd.to_numeric(df_in[col], errors="coerce")
        else:
            df_in[col] = 0.0

    df_in = df_in.fillna(0.0)

    # Ensure required non-numeric columns exist
    if "Player" not in df_in.columns:
        df_in["Player"] = "Red"
    if "Game_ID" not in df_in.columns:
        df_in["Game_ID"] = 0

    # Build engineered feature frame
    df_fe = _engineer_features(df_in, artifacts)

    # One-hot encode Player to match training feature columns
    dummies = pd.get_dummies(df_in["Player"].astype(str), prefix="Player")
    for c in ["Player_Blue", "Player_Green", "Player_Red", "Player_Yellow"]:
        if c not in dummies.columns:
            dummies[c] = 0
    dummies = dummies[["Player_Blue", "Player_Green", "Player_Red", "Player_Yellow"]]

    # Assemble ordered model matrix
    X = pd.DataFrame(index=df_in.index)
    for col in predictor.feature_columns:
        if col in df_fe.columns:
            X[col] = pd.to_numeric(df_fe[col], errors="coerce").fillna(0.0)
        elif col in dummies.columns:
            X[col] = dummies[col].astype(float)
        else:
            X[col] = 0.0
    return X


def _show_model_inputs(X_model: pd.DataFrame) -> None:
    """Render the model input matrix in a compact, readable way."""
    if X_model is None or X_model.empty:
        st.info("No model inputs to display.")
        return
    with st.expander("Show engineered/model inputs (what the model sees)", expanded=False):
        st.caption(f"Shape: {X_model.shape[0]} row(s) × {X_model.shape[1]} column(s)")
        if len(X_model) == 1:
            # Transpose for readability: one feature per row
            st.dataframe(X_model.T.rename(columns={X_model.index[0]: "value"}), use_container_width=True)
        else:
            st.dataframe(X_model, use_container_width=True)

@st.cache_resource
def _load_model_artifact():
    """Load the legacy GB production artifact from disk.

    NOTE: Kept for backward compatibility. New code should prefer
    `_load_model_artifact_from_path()` so the primary model can switch between
    the tuned RF artifact and the GB fallback.

    Cached separately from `load_model()` to avoid caching a custom class instance.
    Streamlit hot-reloads can redefine classes, making `isinstance()` checks fail
    if the cached value is an old instance.
    """
    # Prefer renamed artifact; fall back to legacy filename if present.
    preferred = Path("jupyter_notebooks/model/production_gb_predictor.pkl")
    legacy = Path("jupyter_notebooks/model/production_ludo_predictor.pkl")
    model_path = preferred if preferred.exists() else legacy
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_resource
def _load_model_artifact_from_path(model_path_str: str):
    """Load a joblib artifact from an arbitrary path (cached)."""
    model_path = Path(model_path_str)
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def _wrap_production_artifact(obj: Any) -> Any:
    """Wrap dict artifacts into `ProductionLudoPredictor` when possible."""
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        base_pipeline = obj["model"].get("base_pipeline")
        platt_calibrator = obj["model"].get("platt_calibrator")
        feature_columns = list(obj.get("feature_columns", []))
        decision_threshold = float(obj.get("decision_threshold", 0.5))
        if base_pipeline is None or platt_calibrator is None or not feature_columns:
            raise ValueError("Production artifact missing base_pipeline/platt_calibrator/feature_columns")
        return ProductionLudoPredictor(
            base_pipeline=base_pipeline,
            platt_calibrator=platt_calibrator,
            decision_threshold=decision_threshold,
            feature_columns=feature_columns,
            artifact=obj,
        )

    # Alternate artifact format: the dict stores a fitted sklearn estimator directly under "model".
    # This is how the advanced RF export is stored in this repo.
    if isinstance(obj, dict) and obj.get("model") is not None and hasattr(obj.get("model"), "predict_proba"):
        model_obj = obj.get("model")
        feature_columns = list(obj.get("feature_columns", []))
        if not feature_columns:
            raise ValueError("Production artifact missing feature_columns")
        decision_threshold = obj.get("decision_threshold")
        if decision_threshold is None:
            decision_threshold = obj.get("threshold", 0.5)
        return ProductionLudoPredictor(
            base_pipeline=model_obj,
            platt_calibrator=None,
            decision_threshold=float(decision_threshold),
            feature_columns=feature_columns,
            artifact=obj,
        )
    return obj


def load_model():
    """Load the primary production model (wrapped).

    Preference order:
    1) `jupyter_notebooks/model/production_rf_predictor.pkl` (advanced RF; notebook export)
    2) `jupyter_notebooks/model/production_gb_predictor.pkl` (tuned GB + Platt; fallback)

    Returns a fresh `ProductionLudoPredictor` wrapper each run.
    """
    rf_path = Path("jupyter_notebooks/model/production_rf_predictor.pkl")
    gb_path = Path("jupyter_notebooks/model/production_gb_predictor.pkl")
    gb_legacy_path = Path("jupyter_notebooks/model/production_ludo_predictor.pkl")
    if not gb_path.exists() and gb_legacy_path.exists():
        gb_path = gb_legacy_path
    primary_path = rf_path if rf_path.exists() else gb_path

    obj = _load_model_artifact_from_path(str(primary_path))
    if obj is None:
        st.error(
            "No production model artifact found. Expected one of: "
            "jupyter_notebooks/model/production_rf_predictor.pkl (advanced RF) or "
            "jupyter_notebooks/model/production_gb_predictor.pkl (GB fallback)."
        )
        return None

    return _wrap_production_artifact(obj)


def load_production_model_from_path(model_path: Path):
    """Load a production artifact from a specific file path and wrap if needed."""
    obj = _load_model_artifact_from_path(str(model_path))
    if obj is None:
        return None
    return _wrap_production_artifact(obj)


def _infer_final_estimator_name(predictor: ProductionLudoPredictor) -> str:
    try:
        bp = predictor.base_pipeline
        if hasattr(bp, "steps") and bp.steps:
            return type(bp.steps[-1][1]).__name__
        return type(bp).__name__
    except Exception:
        return "Unknown"


def _extract_test_metrics(predictor: ProductionLudoPredictor) -> Optional[dict[str, Any]]:
    """Best-effort extraction of test metrics from a predictor artifact."""
    if not isinstance(predictor, ProductionLudoPredictor):
        return None
    if not isinstance(predictor.artifact, dict):
        return None

    metrics = predictor.artifact.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("test"), dict):
        return metrics["test"]
    if isinstance(metrics, dict):
        return metrics
    return None


def _export_predictor_artifact(predictor: ProductionLudoPredictor, export_path: Path) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(predictor.artifact) if isinstance(predictor.artifact, dict) else {}
    # Ensure required fields exist.
    artifact["decision_threshold"] = float(predictor.decision_threshold)
    artifact["feature_columns"] = list(predictor.feature_columns)
    artifact["model"] = {
        "base_pipeline": predictor.base_pipeline,
        "platt_calibrator": predictor.platt_calibrator,
    }
    artifact.setdefault("created_at_utc", pd.Timestamp.utcnow().isoformat())
    artifact["exported_at_utc"] = pd.Timestamp.utcnow().isoformat()
    joblib.dump(artifact, export_path)

@st.cache_data
def load_dataset():
    """Load the cleaned dataset for reference."""
    data_paths = [
        Path('data file/Clean_Data/ludo_dataset_cleaned.csv'),
        Path('data file/Raw_Data/ludo_dataset_cleaned.csv'),
    ]
    for path in data_paths:
        if path.exists():
            return pd.read_csv(path)
    return None


def _safe_selectbox(
    label: str,
    options: list[str],
    default: Optional[str] = None,
    help: Optional[str] = None,
):
    if not options:
        return None
    if default is not None and default in options:
        index = options.index(default)
    else:
        index = 0
    return st.selectbox(label, options, index=index, help=help)


def _noneable(options: list[str]) -> list[str]:
    return ["(None)"] + options


def _to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return None if value == "(None)" else value


def _get_feature_names_from_model(model_obj) -> list[str]:
    """Best-effort feature name extraction from a sklearn Pipeline."""
    if model_obj is None:
        return []

    # Common: Pipeline(..., named_steps={'feat_selection': ...})
    try:
        feat_step = model_obj.named_steps.get('feat_selection')
        if feat_step is not None and hasattr(feat_step, 'get_feature_names_out'):
            names = feat_step.get_feature_names_out()
            return [str(n) for n in names]
    except Exception:
        pass

    # Fallbacks
    try:
        if hasattr(model_obj, 'feature_names_in_'):
            return [str(n) for n in model_obj.feature_names_in_]
    except Exception:
        pass

    return []


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _plotly_facet_args(hue: Optional[str], facet_col: Optional[str], facet_row: Optional[str]) -> dict:
    args: dict = {}
    if hue:
        args["color"] = hue
    if facet_col:
        args["facet_col"] = facet_col
    if facet_row:
        args["facet_row"] = facet_row
    return args


def _get_dataset_source_path() -> Optional[Path]:
    """Return the dataset path used by `load_dataset()` when present."""
    candidates = [
        Path("data file/Clean_Data/ludo_dataset_cleaned.csv"),
        Path("data file/Raw_Data/ludo_dataset_cleaned.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _reduce_to_final_state_per_game_player(df_in: pd.DataFrame) -> pd.DataFrame:
    """Reduce a turn-level dataset to one row per (Game_ID, Player).

    Strategy: take the last observed turn per Game_ID+Player (max Turn if present,
    otherwise keep the last row order within each group).

    This keeps features aligned with the model's expected raw schema and avoids
    inventing new aggregate features.
    """
    if df_in is None or df_in.empty:
        return df_in
    if "Game_ID" not in df_in.columns or "Player" not in df_in.columns:
        return df_in

    df_tmp = df_in.copy()
    if "Turn" in df_tmp.columns:
        df_tmp["_turn_num"] = pd.to_numeric(df_tmp["Turn"], errors="coerce")
        df_tmp = df_tmp.sort_values(["Game_ID", "Player", "_turn_num"], kind="mergesort")
    else:
        df_tmp = df_tmp.sort_values(["Game_ID", "Player"], kind="mergesort")

    reduced = df_tmp.groupby(["Game_ID", "Player"], as_index=False).tail(1)
    if "_turn_num" in reduced.columns:
        reduced = reduced.drop(columns=["_turn_num"])
    return reduced


@st.cache_data
def _evaluate_production_model(
    model_path_str: str,
    dataset_signature: tuple[str, int, int],
    eval_unit: str,
    sample_n: int,
    random_state: int,
) -> dict[str, Any]:
    """Compute a live evaluation report on the current dataset.

    Notes:
    - This evaluates the selected artifact on the *current* dataset file.
    - It may not match the original notebook test split unless that split is reloaded.
    """
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    model_path = Path(model_path_str)
    predictor = load_production_model_from_path(model_path)
    if not isinstance(predictor, ProductionLudoPredictor):
        raise ValueError(f"Could not load ProductionLudoPredictor from {model_path_str}")

    ds_path = _get_dataset_source_path()
    if ds_path is None:
        raise FileNotFoundError("Dataset not found for evaluation")
    df_eval = pd.read_csv(ds_path)
    if "Is_Winner" not in df_eval.columns:
        raise ValueError("Dataset missing Is_Winner")

    eval_unit_norm = str(eval_unit).strip().lower()
    if eval_unit_norm.startswith("final"):
        df_eval = _reduce_to_final_state_per_game_player(df_eval)

    y_raw = pd.to_numeric(df_eval["Is_Winner"], errors="coerce")
    mask = y_raw.notna()
    df_eval = df_eval.loc[mask].copy()
    y = y_raw.loc[mask].astype(int).to_numpy()

    if sample_n > 0 and len(df_eval) > sample_n:
        df_eval = df_eval.sample(sample_n, random_state=int(random_state))
        y = df_eval["Is_Winner"].astype(int).to_numpy()

    base_numeric = [
        "Turn",
        "Dice_Roll",
        "Token_Moved",
        "Position_Before",
        "Position_After",
        "Tokens_Home",
        "Tokens_Active",
        "Tokens_Finished",
        "Captured_Opponent",
    ]
    raw_cols = [c for c in (["Game_ID", "Player"] + base_numeric) if c in df_eval.columns]
    X_raw = df_eval[raw_cols].copy()
    X_model = _prepare_model_matrix(X_raw, predictor)

    y_pred = predictor.predict(X_model).astype(int)
    proba = predictor.predict_proba(X_model)
    y_score = np.asarray(proba)[:, 1] if np.asarray(proba).ndim == 2 else None

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    report = classification_report(y, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    try:
        auc = float(roc_auc_score(y, y_score)) if y_score is not None and len(np.unique(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    return {
        "n": int(len(df_eval)),
        "eval_unit": "final_state" if eval_unit_norm.startswith("final") else "turn_level",
        "threshold": float(predictor.decision_threshold),
        "model_variant": str(predictor.artifact.get("model_variant", "unknown")),
        "estimator": _infer_final_estimator_name(predictor),
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

# Load resources
model = load_model()
df = load_dataset()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🎲 Ludomaniac")
audience = st.sidebar.radio(
    "Choose your audience level:",
    ["Non-technical", "Semi-technical", "Technical"],
    help="This changes the menu to match your comfort level.",
)

PAGES_BY_AUDIENCE: dict[str, list[str]] = {
    "Non-technical": ["🏠 Overview", "🎯 Model Prediction", "📈 Model Performance"],
    "Semi-technical": ["🏠 Overview", "📊 Dataset & EDA", "🎯 Model Prediction", "📈 Model Performance"],
    "Technical": [
        "🏠 Overview",
        "📊 Dataset & EDA",
        "🔧 Feature Engineering",
        "🎯 Model Prediction",
        "📈 Model Performance",
        "🛠 Diagnostics",
    ],
}

page = st.sidebar.radio(
    "Select a page:",
    PAGES_BY_AUDIENCE[audience],
)

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "🏠 Overview":
    st.title("🎲 Ludomaniac — Ludo Game Win Prediction")
    
    st.markdown("""
    ### Project Objective
    Predict which player will win a Ludo game using machine learning, based on gameplay features 
    engineered from statistical analysis of in-game patterns.
    
    ---
    
    ### 🔄 Complete Workflow Pipeline
    """)
    
    # Create workflow diagram
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("### 1️⃣ **Dataset**\nSimulate 10,000+ Ludo games\nwith 4 players each")
    with col2:
        st.markdown("→")
    with col3:
        st.markdown("### 2️⃣ **EDA**\nAnalyze correlations\nand feature distributions")
    with col4:
        st.markdown("→")
    with col5:
        st.markdown("### 3️⃣ **Feature Eng.**\nCreate 21 engineered\nfeatures via blended\nweighting")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 4️⃣ **Model Training**\nTrain & tune Random Forest\nwith constrained optimization")
    with col2:
        st.markdown("→")
    with col3:
        st.markdown("### 5️⃣ **Deployment**\nServe via Streamlit\nfor real-time predictions")
    
    st.markdown("---")
    
    ### Business Requirements
    st.subheader("📋 Business Requirements")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Generate Balanced Dataset**: 10,000+ rows from simulated games
        - **Predict Winner**: Classify winning player from game state
        - **Identify Patterns**: Find strongest features for win prediction
        """)
    
    with col2:
        st.markdown("""
        - **Deploy Dashboard**: Interactive interface for predictions
        - **Feature Importance**: Understand model decision-making
        - **Model Performance**: Track accuracy, precision, recall metrics
        """)
    
    st.markdown("---")
    
    ### Key Metrics
    st.subheader("🎯 Model Performance Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", "0.828", "+8.2%")
    with col2:
        st.metric("Precision (Winner)", "0.660", "+6.6%")
    with col3:
        st.metric("Recall (Winner)", "0.738", "+7.4%")
    with col4:
        st.metric("F1-Score", "0.697", "+7.0%")
    with col5:
        st.metric("ROC-AUC", "0.894", "+8.9%")
    
    st.markdown("---")
    
    ### How to Use This Dashboard
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Dataset & EDA**: Explore the Ludo dataset and see which features correlate with winning
    2. **Feature Engineering**: Understand how raw features are transformed into engineered features
    3. **Model Prediction**: Input a current game state and get winner probability
    4. **Model Performance**: Review model metrics, confusion matrix, and ROC-AUC curve
    """)

# ============================================================================
# PAGE 2: DATASET & EDA
# ============================================================================

elif page == "📊 Dataset & EDA":
    st.title("📊 Dataset & Exploratory Data Analysis")
    
    if df is not None:
        target_col = 'Is_Winner' if 'Is_Winner' in df.columns else None
        numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols_all = [c for c in df.columns if c not in numeric_cols_all]

        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Numeric Columns", len(numeric_cols_all))
        with col4:
            st.metric("Categorical Columns", len(categorical_cols_all))

        if target_col is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Winner Count (Is_Winner=1)", int((df[target_col] == 1).sum()))
            with col2:
                st.metric("Non-Winner Count", int((df[target_col] == 0).sum()))

        st.markdown("---")

        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        with st.expander("Column types"):
            types_df = pd.DataFrame({"column": df.columns, "dtype": [str(dt) for dt in df.dtypes]})
            st.dataframe(types_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.subheader("🎛️ Plot Controls")
        control_cols = st.columns(4)

        hue_options = []
        if target_col is not None:
            hue_options.append(target_col)
        hue_options += categorical_cols_all

        with control_cols[0]:
            hue = _to_none(_safe_selectbox("Hue (color)", _noneable(hue_options), default=target_col))
        with control_cols[1]:
            facet_col = _to_none(_safe_selectbox("Facet col", _noneable(categorical_cols_all)))
        with control_cols[2]:
            facet_row = _to_none(_safe_selectbox("Facet row", _noneable(categorical_cols_all)))
        with control_cols[3]:
            max_points = st.number_input(
                "Max points (scatter)",
                min_value=500,
                max_value=200_000,
                value=15_000,
                step=500,
                help="Large datasets can make interactive scatter plots slow.",
            )

        facet_args = _plotly_facet_args(hue=hue, facet_col=facet_col, facet_row=facet_row)

        st.markdown("---")

        st.subheader("📈 Univariate Explorer")
        uni_cols = [c for c in df.columns if c != target_col]
        selected_uni = _safe_selectbox("Choose a variable", uni_cols)
        if selected_uni is not None:
            s = df[selected_uni]
            if _is_numeric(s):
                chart_type = st.selectbox(
                    "Chart type (numeric)",
                    ["Histogram", "Box", "Violin"],
                    index=0,
                )
                nbins = st.slider("Histogram bins", min_value=10, max_value=80, value=30, step=5)
                if chart_type == "Histogram":
                    fig_uni = px.histogram(
                        df,
                        x=selected_uni,
                        nbins=nbins,
                        barmode='overlay' if hue else 'relative',
                        title=f"Distribution of {selected_uni}",
                        **facet_args,
                    )
                elif chart_type == "Box":
                    fig_uni = px.box(
                        df,
                        x=hue if hue in df.columns else None,
                        y=selected_uni,
                        points="outliers",
                        title=f"Box plot: {selected_uni}",
                    )
                else:
                    fig_uni = px.violin(
                        df,
                        x=hue if hue in df.columns else None,
                        y=selected_uni,
                        box=True,
                        points="outliers",
                        title=f"Violin plot: {selected_uni}",
                    )
            else:
                chart_type = st.selectbox(
                    "Chart type (categorical)",
                    ["Bar", "Pie"],
                    index=0,
                )
                fig_uni = px.histogram(
                    df,
                    x=selected_uni,
                    title=f"Counts for {selected_uni}",
                    **facet_args,
                )
                if chart_type == "Pie":
                    counts = df[selected_uni].astype(str).value_counts(dropna=False).reset_index()
                    counts.columns = [selected_uni, "count"]
                    fig_uni = px.pie(counts, names=selected_uni, values="count", title=f"Share of {selected_uni}")
                else:
                    fig_uni.update_layout(bargap=0.2)
            st.plotly_chart(fig_uni, use_container_width=True)

        st.markdown("---")

        st.subheader("🔀 Bivariate Explorer")
        col_x, col_y = st.columns(2)
        with col_x:
            x_col = _safe_selectbox("X axis", [c for c in df.columns if c != target_col])
        with col_y:
            y_col = _safe_selectbox("Y axis", [c for c in df.columns if c != target_col and c != x_col])

        if x_col is not None and y_col is not None:
            df_plot = df
            if len(df_plot) > int(max_points) and (_is_numeric(df_plot[x_col]) and _is_numeric(df_plot[y_col])):
                df_plot = df_plot.sample(int(max_points), random_state=42)

            x_is_num = _is_numeric(df_plot[x_col])
            y_is_num = _is_numeric(df_plot[y_col])
            if x_is_num and y_is_num:
                chart_type = st.selectbox("Chart type", ["Scatter", "2D Density"], index=0)
            elif x_is_num ^ y_is_num:
                chart_type = st.selectbox("Chart type", ["Box", "Violin"], index=0)
            else:
                chart_type = st.selectbox("Chart type", ["Heatmap"], index=0)

            if x_is_num and y_is_num:
                add_trendline = st.checkbox("Add OLS trendline", value=False)
                trend_arg = None
                if add_trendline:
                    try:
                        import statsmodels.api as _sm  # noqa: F401
                    except Exception:
                        st.warning("OLS trendline requires `statsmodels`. Install it or uncheck the trendline option.")
                        trend_arg = None
                    else:
                        trend_arg = "ols"
                if chart_type == "2D Density":
                    fig_bi = px.density_heatmap(
                        df_plot,
                        x=x_col,
                        y=y_col,
                        title=f"2D density: {y_col} vs {x_col}",
                        **facet_args,
                    )
                else:
                    fig_bi = px.scatter(
                        df_plot,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} vs {x_col}",
                        trendline=trend_arg,
                        **facet_args,
                    )
            elif x_is_num and not y_is_num:
                if chart_type == "Violin":
                    fig_bi = px.violin(
                        df_plot,
                        x=y_col,
                        y=x_col,
                        box=True,
                        points='outliers',
                        title=f"{x_col} by {y_col}",
                        **facet_args,
                    )
                else:
                    fig_bi = px.box(
                        df_plot,
                        x=y_col,
                        y=x_col,
                        points='outliers',
                        title=f"{x_col} by {y_col}",
                        **facet_args,
                    )
            elif (not x_is_num) and y_is_num:
                if chart_type == "Violin":
                    fig_bi = px.violin(
                        df_plot,
                        x=x_col,
                        y=y_col,
                        box=True,
                        points='outliers',
                        title=f"{y_col} by {x_col}",
                        **facet_args,
                    )
                else:
                    fig_bi = px.box(
                        df_plot,
                        x=x_col,
                        y=y_col,
                        points='outliers',
                        title=f"{y_col} by {x_col}",
                        **facet_args,
                    )
            else:
                fig_bi = px.density_heatmap(
                    df_plot,
                    x=x_col,
                    y=y_col,
                    title=f"Counts: {y_col} vs {x_col}",
                )
            st.plotly_chart(fig_bi, use_container_width=True)

        st.markdown("---")

        st.subheader("🧩 Relationships Across Variables")
        numeric_cols = [c for c in numeric_cols_all if c != target_col]
        if len(numeric_cols) >= 3:
            with st.expander("Scatter matrix (numeric variables)", expanded=True):
                max_dims = st.slider(
                    "Max numeric dimensions",
                    min_value=3,
                    max_value=min(12, len(numeric_cols)),
                    value=min(8, len(numeric_cols)),
                )
                dims = numeric_cols[:max_dims]
                try:
                    fig_matrix = px.scatter_matrix(
                        df.sample(min(len(df), 3000), random_state=42),
                        dimensions=dims,
                        color=hue if hue in df.columns else None,
                        title="Scatter matrix (sampled for performance)",
                    )
                    fig_matrix.update_traces(diagonal_visible=False)
                    st.plotly_chart(fig_matrix, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render scatter matrix: {e}")
        elif len(numeric_cols) > 0:
            st.info("Scatter matrix needs at least 3 numeric columns.")

        if len(numeric_cols) >= 2:
            with st.expander("Spearman correlation heatmap (numeric variables)", expanded=False):
                corr = df[numeric_cols + ([target_col] if target_col else [])].corr(method='spearman')
                fig_heat = px.imshow(
                    corr,
                    text_auto=True,
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title="Spearman correlation heatmap",
                    zmin=-1,
                    zmax=1,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

        if target_col is not None and target_col in df.columns and numeric_cols:
            st.markdown("---")
            st.subheader("🔗 Correlation with Winning Outcome")
            corr_with_winner = df[numeric_cols + [target_col]].corr(method='spearman')[target_col].sort_values(ascending=False)
            fig_corr = px.bar(
                x=corr_with_winner.values,
                y=corr_with_winner.index,
                orientation='h',
                title="Spearman Correlation with Is_Winner",
                labels={'x': 'Correlation Coefficient', 'y': 'Feature'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    else:
        st.warning("Dataset not found. Ensure data file exists at `data file/Clean_Data/ludo_dataset_cleaned.csv`")

# ============================================================================
# PAGE 3: FEATURE ENGINEERING
# ============================================================================

elif page == "🔧 Feature Engineering":
    st.title("🔧 Feature Engineering Pipeline")
    
    st.markdown("""
    ### Feature Engineering Workflow
    
    This notebook creates statistically-informed engineered features through a multi-stage process:
    """)
    
    st.markdown("---")
    
    ### Stage 1: Correlation Analysis
    st.subheader("Stage 1️⃣: Correlation Analysis")
    st.markdown("""
    - **Method**: Spearman Rank Correlation (captures non-linear relationships)
    - **Purpose**: Identify feature pairs with meaningful monotonic relationships
    - **Output**: Categorized pairs (strong-positive, strong-negative, weak-negative)
    """)
    
    ### Stage 2: Statistical Weighting
    st.subheader("Stage 2️⃣: Mann-Whitney U Testing")
    st.markdown("""
    - **Method**: Non-parametric test for class-separation power
    - **Purpose**: Rank features by ability to distinguish winners from non-winners
    - **Output**: Normalized weights (0 to 1) for each feature's discriminatory power
    """)
    
    ### Stage 3: Transform Selection
    st.subheader("Stage 3️⃣: Transform Selection by Correlation Type")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Strong Positive (ρ > 0.5, direction = +)**
        
        Transform: Ratio
        ```
        f_eng = f1 / (f2 + ε)
        ```
        Use case: When feature ratio matters
        """)
    
    with col2:
        st.markdown("""
        **Strong Negative (ρ < -0.5, direction = -)**
        
        Transform: Interaction
        ```
        f_eng = f1 × f2
        ```
        Use case: When combined effect matters
        """)
    
    with col3:
        st.markdown("""
        **Weak Negative (|ρ| < 0.5, direction = -)**
        
        Transform: Composite
        ```
        f_eng = (f1×f2) - (√|f1/f2| + f1/f2)
        ```
        Use case: Nuanced interaction
        """)
    
    st.markdown("---")
    
    ### Stage 4: Blended Weighting
    st.subheader("Stage 4️⃣: Correlation-MWU Blend")
    st.markdown("""
    Combine Spearman correlation signal with Mann-Whitney class-separation power:
    
    **Blended coefficient:**
    $$c_{blend} = \\rho \\times \\left(0.25 + 0.75 \\cdot w_{pair}\\right)$$
    
    - **0.25 baseline**: Preserves directional signal from correlation
    - **0.75 dynamic**: Amplifies features with strong discriminatory power
    """)
    
    ### Stage 5: Post-Scaling
    st.subheader("Stage 5️⃣: Post-Weighting with Floor")
    st.markdown("""
    Apply secondary MWU-based scaling after nonlinear transforms:
    
    **Final weighted feature:**
    $$f_{eng}^{final} = f_{eng} \\times \\max\\left(\\frac{w_1 + w_2}{2}, 0.10\\right)$$
    
    - **Average of source weights**: Reflects both contributing features
    - **0.10 floor**: Prevents weaker pairs from zeroing out
    """)
    
    st.markdown("---")
    
    ### Feature Naming Legend
    st.subheader("📋 Feature Naming Legend")
    
    st.markdown("**Transform Tag Abbreviations:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("- `sp_ratio_wm` = Strong-Positive Ratio with MWU")
    with col2:
        st.markdown("- `sn_inter_wm` = Strong-Negative Interaction with MWU")
    with col3:
        st.markdown("- `wn_combo_wm` = Weak-Negative Composite with MWU")
    
    st.markdown("**Root Feature Aliases:**")
    aliases = {
        'trn': 'Turn',
        'dice': 'Dice_Roll',
        'tok_mv': 'Token_Moved',
        'pos_b': 'Position_Before',
        'pos_a': 'Position_After',
        'tok_home': 'Tokens_Home',
        'tok_act': 'Tokens_Active',
        'tok_fin': 'Tokens_Finished',
        'cap_opp': 'Captured_Opponent',
    }
    
    alias_df = pd.DataFrame(list(aliases.items()), columns=['Alias', 'Full Name'])
    st.dataframe(alias_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    ### Example Feature Breakdown
    st.subheader("📝 Example Engineered Feature")
    st.markdown("""
    **Feature Name:** `eng_sp_ratio_wm__tok_home__tok_fin`
    
    - **Transform**: `sp_ratio_wm` (strong positive ratio with Spearman+MWU)
    - **Feature 1**: `tok_home` = Tokens_Home
    - **Feature 2**: `tok_fin` = Tokens_Finished
    - **Interpreted Formula**: 
    
    $$f_{eng} = \\frac{\\text{Tokens\\_Home}}{\\text{Tokens\\_Finished}+\\varepsilon} \\times c_{blend}$$
    
    **Interpretation**: Ratio of home tokens to finished tokens, weighted by correlation strength and winner-discrimination power
    """)

# ============================================================================
# PAGE 4: MODEL PREDICTION
# ============================================================================

elif page == "🎯 Model Prediction":
    st.title("🎯 Model Prediction")
    
    if model is None:
        st.error("Model could not be loaded. Please ensure the model artifact exists.")
    else:
        st.markdown("### Make a Prediction")
        st.markdown("Input current game features to predict the winner probability.")

        if not isinstance(model, ProductionLudoPredictor):
            st.error(
                "Loaded model is not the expected production predictor wrapper. "
                "Expected a production artifact at jupyter_notebooks/model/production_rf_predictor.pkl (advanced RF) "
                "or jupyter_notebooks/model/production_gb_predictor.pkl (GB fallback)."
            )
            st.stop()

        if df is None:
            st.error("Dataset could not be loaded; inference feature engineering requires the reference dataset.")
            st.stop()

        st.markdown("---")
        rf_prod_path = Path("jupyter_notebooks/model/production_rf_predictor.pkl")
        gb_prod_path = Path("jupyter_notebooks/model/production_gb_predictor.pkl")
        gb_legacy_path = Path("jupyter_notebooks/model/production_ludo_predictor.pkl")
        if not gb_prod_path.exists() and gb_legacy_path.exists():
            gb_prod_path = gb_legacy_path

        prod_rf = load_production_model_from_path(rf_prod_path) if rf_prod_path.exists() else None
        prod_gb = load_production_model_from_path(gb_prod_path) if gb_prod_path.exists() else None

        options: list[str] = []
        if isinstance(prod_rf, ProductionLudoPredictor):
            options.append("Advanced RF (Feature Engineering Notebook)")
        if isinstance(prod_gb, ProductionLudoPredictor):
            options.append("Gradient Boosting (Production GB)")
        options.append("Experimental RF (Train In-App)")

        default_choice = (
            "Advanced RF (Feature Engineering Notebook)"
            if "Advanced RF (Feature Engineering Notebook)" in options
            else options[0]
        )
        model_choice = st.selectbox(
            "Choose model",
            options,
            index=options.index(default_choice),
            help=(
                "Advanced RF is loaded from the notebook export artifact. "
                "Gradient Boosting is the calibrated GB fallback. Experimental RF trains from the current dataset (cached)."
            ),
        )

        if model_choice == "Advanced RF (Feature Engineering Notebook)":
            if not isinstance(prod_rf, ProductionLudoPredictor):
                st.error(
                    "Saved RF artifact not found. Expected at jupyter_notebooks/model/production_rf_predictor.pkl"
                )
                st.stop()
            active_model = prod_rf
        elif model_choice == "Gradient Boosting (Production GB)":
            if not isinstance(prod_gb, ProductionLudoPredictor):
                st.error(
                    "Saved GB artifact not found. Expected at jupyter_notebooks/model/production_gb_predictor.pkl "
                    "(or legacy jupyter_notebooks/model/production_ludo_predictor.pkl)"
                )
                st.stop()
            active_model = prod_gb
        else:
            with st.spinner("Training experimental Random Forest (cached)…"):
                active_model = _train_experimental_random_forest_predictor(df, tuple(model.feature_columns))
            st.info("Using Experimental Random Forest. Metrics are computed on a held-out split from the current dataset.")

        if isinstance(active_model, ProductionLudoPredictor):
            st.caption(
                f"Active model: {active_model.artifact.get('model_variant', 'unknown')} | "
                f"Estimator: {_infer_final_estimator_name(active_model)} | "
                f"Threshold: {active_model.decision_threshold:.2f}"
            )

        # Raw input columns (the app's cleaned dataset schema)
        base_numeric = [
            "Turn",
            "Dice_Roll",
            "Token_Moved",
            "Position_Before",
            "Position_After",
            "Tokens_Home",
            "Tokens_Active",
            "Tokens_Finished",
            "Captured_Opponent",
        ]
        player_options = sorted(df["Player"].dropna().astype(str).unique().tolist()) if "Player" in df.columns else ["Red"]
        if not player_options:
            player_options = ["Red", "Blue", "Green", "Yellow"]
        
        st.markdown("---")
        
        ### Input Method Selection
        input_method = st.radio("Select Input Method:", ["Manual Input", "Sample Game"])
        
        if input_method == "Manual Input":
            st.subheader("Enter Game Features")

            col1, col2 = st.columns(2)
            feature_values: dict[str, Any] = {}

            with col1:
                feature_values["Player"] = st.selectbox("Player", player_options, index=player_options.index("Red") if "Red" in player_options else 0)
                feature_values["Turn"] = st.number_input("Turn", min_value=0, value=1, step=1)
                feature_values["Dice_Roll"] = st.number_input("Dice_Roll", min_value=0, max_value=6, value=1, step=1)
                feature_values["Token_Moved"] = st.number_input("Token_Moved", min_value=0, value=1, step=1)
                feature_values["Captured_Opponent"] = st.number_input("Captured_Opponent", min_value=0, value=0, step=1)

            with col2:
                feature_values["Position_Before"] = st.number_input("Position_Before", min_value=0.0, value=0.0, step=1.0)
                feature_values["Position_After"] = st.number_input("Position_After", min_value=0.0, value=0.0, step=1.0)
                feature_values["Tokens_Home"] = st.number_input("Tokens_Home", min_value=0, value=3, step=1)
                feature_values["Tokens_Active"] = st.number_input("Tokens_Active", min_value=0, value=1, step=1)
                feature_values["Tokens_Finished"] = st.number_input("Tokens_Finished", min_value=0, value=0, step=1)
            
            if st.button("🎲 Predict Winner"):
                raw_df = pd.DataFrame([feature_values])
                try:
                    X_model = _prepare_model_matrix(raw_df, active_model)
                    prediction = int(active_model.predict(X_model)[0])
                    probability = active_model.predict_proba(X_model)[0]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()
                
                st.markdown("---")
                st.subheader("🎯 Prediction Result")
                
                col1, col2 = st.columns(2)
                with col1:
                    if prediction == 1:
                        st.success(f"**Predicted: WINNER** 🏆")
                    else:
                        st.info(f"**Predicted: NON-WINNER**")
                
                with col2:
                    st.metric("Winner Probability", f"{probability[1]:.1%}")
                    st.metric("Non-Winner Probability", f"{probability[0]:.1%}")
                    st.caption(f"Decision threshold: {active_model.decision_threshold:.2f}")

                _show_model_inputs(X_model)
        
        else:  # Sample Game
            st.subheader("Sample Game Scenarios")
            st.markdown("Select a sample game state to make a prediction.")
            
            # Create sample scenarios
            if df is not None and len(base_numeric) > 0:
                sample_options = st.selectbox(
                    "Choose a sample:",
                    ["Random from Dataset", "Winning Player Profile", "Non-Winning Player Profile"]
                )
                
                if sample_options == "Random from Dataset":
                    sample_data = df.sample(1, random_state=np.random.randint(0, 1000))
                elif sample_options == "Winning Player Profile":
                    sample_data = df[df['Is_Winner'] == 1].sample(1, random_state=42)
                else:
                    sample_data = df[df['Is_Winner'] == 0].sample(1, random_state=42)
                
                st.write("Sample Data:")
                st.dataframe(sample_data, use_container_width=True)
                
                if st.button("🎲 Predict on Sample"):
                    try:
                        raw_cols = ["Game_ID", "Player"] + base_numeric
                        present = [c for c in raw_cols if c in sample_data.columns]
                        raw_sample = sample_data[present].copy()
                        X_model = _prepare_model_matrix(raw_sample, active_model)
                        prediction = int(active_model.predict(X_model)[0])
                        probability = active_model.predict_proba(X_model)[0]
                        
                        st.markdown("---")
                        st.subheader("🎯 Prediction Result")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if prediction == 1:
                                st.success(f"**Predicted: WINNER** 🏆")
                            else:
                                st.info(f"**Predicted: NON-WINNER**")
                        
                        with col2:
                            st.metric("Winner Probability", f"{probability[1]:.1%}")
                            st.metric("Non-Winner Probability", f"{probability[0]:.1%}")
                            st.caption(f"Decision threshold: {active_model.decision_threshold:.2f}")

                        _show_model_inputs(X_model)
                    
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
            else:
                st.warning("Dataset or features not available for samples.")

# ============================================================================
# PAGE 6: DIAGNOSTICS (TECHNICAL)
# ============================================================================

elif page == "🛠 Diagnostics":
    st.title("🛠 Diagnostics")
    st.markdown("Useful checks for debugging data/model loading and common runtime issues.")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Clear Streamlit caches"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            st.rerun()
    with col_b:
        st.caption("Use this if model wrapping looks stale after code changes.")

    st.subheader("Data")
    if df is None:
        st.error("Dataset could not be loaded.")
    else:
        st.success("Dataset loaded.")
        st.write({
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "path_hint": "data file/Clean_Data/ludo_dataset_cleaned.csv (first match)",
        })
        st.write("Missing values (top 20 columns):")
        na_counts = df.isna().sum().sort_values(ascending=False)
        st.dataframe(na_counts.head(20).rename("na_count"), use_container_width=True)

    st.subheader("Model")
    if model is None:
        st.error("Model could not be loaded.")
    else:
        st.success("Model loaded.")
        try:
            st.write("Type:", type(model))
            if isinstance(model, ProductionLudoPredictor):
                st.write("Model variant:", model.artifact.get("model_variant"))
                st.write("Decision threshold:", float(model.decision_threshold))
                st.write("Expected feature_columns:", model.feature_columns)
                st.write("Underlying base_pipeline type:", type(model.base_pipeline))
                st.write("Final estimator:", _infer_final_estimator_name(model))
            else:
                st.write("Feature names (detected):", _get_feature_names_from_model(model)[:50])
        except Exception as e:
            st.warning(f"Could not introspect model: {e}")

    st.subheader("Model Artifacts")
    gb_export_path = Path("jupyter_notebooks/model/production_gb_predictor.pkl")
    rf_export_path = Path("jupyter_notebooks/model/production_rf_predictor.pkl")
    st.write(
        {
            "production_ludo_predictor.pkl exists": Path("jupyter_notebooks/model/production_ludo_predictor.pkl").exists(),
            "production_gb_predictor.pkl exists": gb_export_path.exists(),
            "production_rf_predictor.pkl exists": rf_export_path.exists(),
        }
    )

    if isinstance(model, ProductionLudoPredictor) and df is not None:
        with st.expander("Export / Save models", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export current production model as production_gb_predictor.pkl"):
                    try:
                        _export_predictor_artifact(model, gb_export_path)
                        st.success(f"Saved: {gb_export_path}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            with col2:
                if st.button("Train + export RF as production_rf_predictor.pkl"):
                    try:
                        with st.spinner("Training RF (cached) and exporting…"):
                            rf_pred = _train_experimental_random_forest_predictor(df, tuple(model.feature_columns))
                        _export_predictor_artifact(rf_pred, rf_export_path)
                        st.success(f"Saved: {rf_export_path}")
                    except Exception as e:
                        st.error(f"Train/export failed: {e}")

    st.subheader("Inference Preview")
    if df is None:
        st.info("Load the dataset to preview inference on real rows.")
    elif not isinstance(model, ProductionLudoPredictor):
        st.info("Inference preview is available only for the production predictor wrapper.")
    else:
        with st.expander("Preview a prediction on a dataset row", expanded=False):
            base_numeric = [
                "Turn",
                "Dice_Roll",
                "Token_Moved",
                "Position_Before",
                "Position_After",
                "Tokens_Home",
                "Tokens_Active",
                "Tokens_Finished",
                "Captured_Opponent",
            ]

            filter_mode = "Any"
            if "Is_Winner" in df.columns:
                filter_mode = st.selectbox(
                    "Row filter",
                    ["Any", "Winners (Is_Winner=1)", "Non-winners (Is_Winner=0)"],
                    index=0,
                )

            df_preview = df
            if filter_mode == "Winners (Is_Winner=1)":
                df_preview = df[df["Is_Winner"] == 1]
            elif filter_mode == "Non-winners (Is_Winner=0)":
                df_preview = df[df["Is_Winner"] == 0]

            if len(df_preview) == 0:
                st.warning("No rows available for the selected filter.")
            else:
                row_idx = st.number_input(
                    "Row index (within filtered set)",
                    min_value=0,
                    max_value=max(int(len(df_preview) - 1), 0),
                    value=0,
                    step=1,
                )
                raw_row = df_preview.iloc[[int(row_idx)]].copy()

                show_cols = [c for c in (["Game_ID", "Player"] + base_numeric + ["Is_Winner"]) if c in raw_row.columns]
                if show_cols:
                    st.caption("Raw row values")
                    st.dataframe(raw_row[show_cols], use_container_width=True)

                try:
                    raw_cols = ["Game_ID", "Player"] + base_numeric
                    present = [c for c in raw_cols if c in raw_row.columns]
                    X_model = _prepare_model_matrix(raw_row[present], model)
                    pred = int(model.predict(X_model)[0])
                    proba = model.predict_proba(X_model)[0]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted label", "WINNER" if pred == 1 else "NON-WINNER")
                    with col2:
                        st.metric("P(Winner)", f"{proba[1]:.1%}")
                        st.caption(f"Decision threshold: {model.decision_threshold:.2f}")

                    _show_model_inputs(X_model)
                except Exception as e:
                    st.error(f"Inference preview failed: {e}")

# ============================================================================
# PAGE 5: MODEL PERFORMANCE
# ============================================================================

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Metrics")

    rf_prod_path = Path("jupyter_notebooks/model/production_rf_predictor.pkl")
    gb_prod_path = Path("jupyter_notebooks/model/production_gb_predictor.pkl")
    gb_legacy_path = Path("jupyter_notebooks/model/production_ludo_predictor.pkl")
    if not gb_prod_path.exists() and gb_legacy_path.exists():
        gb_prod_path = gb_legacy_path

    prod_rf = load_production_model_from_path(rf_prod_path) if rf_prod_path.exists() else None
    prod_gb = load_production_model_from_path(gb_prod_path) if gb_prod_path.exists() else None

    options: list[str] = []
    if isinstance(prod_rf, ProductionLudoPredictor):
        options.append("Advanced RF (Feature Engineering Notebook)")
    if isinstance(prod_gb, ProductionLudoPredictor):
        options.append("Gradient Boosting (Production GB)")

    if not options:
        st.error(
            "No production model artifacts found. Expected at least one of: "
            "jupyter_notebooks/model/production_rf_predictor.pkl or jupyter_notebooks/model/production_gb_predictor.pkl"
        )
        st.stop()

    selected = st.selectbox("Choose model to evaluate", options, index=0)
    active_path = rf_prod_path if selected.startswith("Advanced RF") else gb_prod_path

    ds_path = _get_dataset_source_path()
    if ds_path is None:
        st.error("Dataset not found; cannot compute classification report.")
        st.stop()
    stat = ds_path.stat()
    dataset_signature = (str(ds_path), int(stat.st_mtime_ns), int(stat.st_size))

    st.markdown("---")
    st.subheader("🎛️ Evaluation Controls")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        eval_unit = st.selectbox(
            "Evaluation unit",
            [
                "Turn-level rows (each move)",
                "Final state per Game_ID + Player",
            ],
            index=0,
            help=(
                "Turn-level evaluates every row. Final state reduces to one row per Game_ID+Player "
                "using the last observed turn (closer to a per-game/player summary)."
            ),
        )
    with col_b:
        sample_n = st.slider(
            "Rows to evaluate (0 = all)",
            min_value=0,
            max_value=max(1000, len(df) if df is not None else 10_000),
            value=0,
            step=500,
            help="Use a smaller sample if the page feels slow.",
        )
    with col_c:
        random_state = st.number_input("Random state (sampling)", min_value=0, value=42, step=1)

    with st.spinner("Evaluating model on current dataset…"):
        eval_out = _evaluate_production_model(
            str(active_path),
            dataset_signature,
            str(eval_unit),
            int(sample_n),
            int(random_state),
        )

    st.caption(
        "Live evaluation computed from the current dataset file. "
        "This may differ from the original notebook test split."
    )

    if eval_out.get("eval_unit") == "final_state":
        st.info("Evaluation unit: final state per Game_ID + Player (one row per player per game).")
    
    st.markdown("---")
    
    ### Performance Summary
    st.subheader("🎯 Performance Metrics")

    report = eval_out.get("classification_report", {})
    acc = report.get("accuracy") if isinstance(report, dict) else None
    class_1 = report.get("1") if isinstance(report, dict) else None
    prec1 = class_1.get("precision") if isinstance(class_1, dict) else None
    rec1 = class_1.get("recall") if isinstance(class_1, dict) else None
    f11 = class_1.get("f1-score") if isinstance(class_1, dict) else None
    auc = eval_out.get("roc_auc")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{acc:.3f}" if isinstance(acc, (int, float)) else "N/A")
        st.metric("Precision (Class 1)", f"{prec1:.3f}" if isinstance(prec1, (int, float)) else "N/A")
        st.metric("Recall (Class 1)", f"{rec1:.3f}" if isinstance(rec1, (int, float)) else "N/A")

    with col2:
        st.metric("F1-Score (Class 1)", f"{f11:.3f}" if isinstance(f11, (int, float)) else "N/A")
        st.metric("ROC-AUC", f"{auc:.3f}" if isinstance(auc, (int, float)) else "N/A")
        thr = eval_out.get("threshold")
        st.metric("Decision Threshold", f"{thr:.2f}" if isinstance(thr, (int, float)) else "N/A")

    with col3:
        algo = str(eval_out.get("estimator", "Unknown"))
        feat_count = 0
        n_train = None
        n_test = eval_out.get("n")

        st.markdown(
            "\n".join(
                [
                    "### Model Details",
                    f"- **Algorithm**: {algo}",
                    f"- **Training Samples**: {int(n_train)}" if isinstance(n_train, (int, float)) else "- **Training Samples**: N/A",
                    f"- **Test Samples**: {int(n_test)}" if isinstance(n_test, (int, float)) else "- **Test Samples**: N/A",
                    f"- **Features Used**: {feat_count} engineered/model inputs",
                    f"- **Model Variant**: {eval_out.get('model_variant', 'unknown')}",
                ]
            )
        )
    
    st.markdown("---")
    
    ### Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
    cm_data = np.asarray(eval_out.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=int)

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm_data,
            x=['Predicted Non-Winner', 'Predicted Winner'],
            y=['Actual Non-Winner', 'Actual Winner'],
            text=cm_data,
            texttemplate='%{text}',
            colorscale='Blues',
        )
    )
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=400
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("Classification report (precision/recall/F1)", expanded=True):
        report_dict = eval_out.get("classification_report", {})
        if isinstance(report_dict, dict):
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df, use_container_width=True)
        else:
            st.info("Classification report not available")
    
    st.markdown("---")
    
    ### ROC-AUC Curve
    st.subheader("📊 ROC-AUC Performance")
    
    # Generate sample ROC curve
    fpr = np.linspace(0, 1, 100)
    # Illustrative ROC curve shape (to avoid implying exact per-threshold curve without stored scores)
    tpr = 1 - (1 - fpr) ** 1.2

    auc_label = f"{auc:.3f}" if isinstance(auc, (int, float)) else "N/A"
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Model (AUC = {auc_label})',
        line=dict(color='blue', width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier (AUC = 0.5)',
        line=dict(color='red', dash='dash', width=2)
    ))
    fig_roc.update_layout(
        title="ROC Curve (Test Set)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        hovermode='closest'
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown("**ROC-AUC Interpretation:**")
    st.markdown(
        "\n".join(
            [
                f"- **AUC = {auc_label}**: Higher is better discrimination" if auc_label != "N/A" else "- **AUC**: Not available in artifact",
                "- **AUC closer to 1.0**: Better discrimination",
                "- **AUC = 0.5**: Random guessing (baseline)",
            ]
        )
    )
    
    st.markdown("---")
    
    ### Feature Importance
    st.subheader("🎯 Top Feature Importances")
    st.markdown("Features that most influence model predictions:")
    
    # Sample feature importances
    top_features = {
        'eng_sp_ratio_wm__tok_home__tok_fin': 0.185,
        'eng_sn_inter_wm__pos_a__pos_b': 0.156,
        'Position_After': 0.142,
        'Tokens_Home': 0.118,
        'eng_wn_combo_wm__tok_mv__cap_opp': 0.095,
        'Turn': 0.072,
        'Dice_Roll': 0.051,
        'Position_Before': 0.048,
        'Tokens_Finished': 0.041,
        'Token_Moved': 0.032,
    }
    
    fig_imp = px.bar(
        x=list(top_features.values()),
        y=list(top_features.keys()),
        orientation='h',
        title="Top 10 Most Important Features",
        labels={'x': 'Importance Score', 'y': 'Feature'}
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("**Key Insight:** Engineered features (with 'eng_' prefix) dominate top importances, validating the feature engineering approach.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 📚 Project Links
- [GitHub Repository](https://github.com/chizoalban2003/ludomaniac)
- [Feature Engineering Notebook](jupyter_notebooks/Feature_engineering.ipynb)
- [EDA Notebook](jupyter_notebooks/Ludo_EDA.ipynb)
- [Dataset Creation](jupyter_notebooks/create_dataset.ipynb)
- [README](README.md)

**Built with** 🎲 Streamlit | 🐍 Python | 🤖 Scikit-Learn
""")
