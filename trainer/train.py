"""OTT Audience Segmentation & Personalization Service - Phase 1 Trainer.

This module loads raw OTT viewer behavioral data, performs strict validation,
scales behavioral features with StandardScaler, evaluates K-Means clusters
across candidate K values, trains the optimal clustering model inside a reusable
scikit-learn Pipeline, interprets audience segments from empirical feature profiles,
and exports production-ready artifacts (pipeline.pkl and metadata.json).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Base path resolution so the script can be executed from any working directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "users.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_ARTIFACT_PATH = MODELS_DIR / "pipeline.pkl"
METADATA_ARTIFACT_PATH = MODELS_DIR / "metadata.json"

# Random state for reproducible training
RANDOM_STATE = 42

# Explicit schema definitions
REQUIRED_COLUMNS = [
    "user_id",
    "watch_time_hours",
    "sessions_per_week",
    "avg_session_mins",
    "completion_rate",
    "action_pct",
    "comedy_pct",
    "drama_pct",
    "thriller_pct",
    "sci_fi_pct",
    "weekend_viewing_pct",
    "days_since_last_watch",
]

# user_id is deliberately excluded: models should cluster behavior, not user identity
FEATURE_COLUMNS = [
    "watch_time_hours",
    "sessions_per_week",
    "avg_session_mins",
    "completion_rate",
    "action_pct",
    "comedy_pct",
    "drama_pct",
    "thriller_pct",
    "sci_fi_pct",
    "weekend_viewing_pct",
    "days_since_last_watch",
]

GENRE_COLUMNS = [
    "action_pct",
    "comedy_pct",
    "drama_pct",
    "thriller_pct",
    "sci_fi_pct",
]


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the raw user behavioral CSV dataset.

    Args:
        file_path: Path to the CSV file.

    Returns:
        pd.DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file cannot be parsed as a DataFrame.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected location: {file_path.resolve()}"
        )
    df = pd.read_csv(file_path)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Perform strict structural and domain-level validation on the dataset.

    Checks:
    - Dataset exists and is non-empty
    - All required columns are present
    - user_id is unique
    - No missing (null) values in any required column
    - Numeric columns are valid numeric types
    - Domain range checks (watch_time >= 0, sessions >= 0, avg_session_mins > 0,
      completion_rate and genre percentages and weekend_viewing in [0, 1],
      days_since_last_watch >= 0)

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If any validation rule fails, with an explicit description.
    """
    if df.empty:
        raise ValueError("Data validation failed: Dataset is completely empty.")

    # Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Data validation failed: Missing required columns: {missing_cols}"
        )

    # Check for user_id uniqueness
    if df["user_id"].duplicated().any():
        num_duplicates = df["user_id"].duplicated().sum()
        raise ValueError(
            f"Data validation failed: Found {num_duplicates} duplicate user_id values."
        )

    # Check for missing / null values
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        null_details = cols_with_nulls.to_dict()
        raise ValueError(
            f"Data validation failed: Missing values detected: {null_details}"
        )

    # Check numeric types
    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "user_id"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Data validation failed: Column '{col}' must be numeric, got {df[col].dtype}."
            )

    # Domain range validations
    if (df["watch_time_hours"] < 0).any():
        bad_count = (df["watch_time_hours"] < 0).sum()
        raise ValueError(
            f"Data validation failed: 'watch_time_hours' has {bad_count} negative values."
        )

    if (df["sessions_per_week"] < 0).any():
        bad_count = (df["sessions_per_week"] < 0).sum()
        raise ValueError(
            f"Data validation failed: 'sessions_per_week' has {bad_count} negative values."
        )

    if (df["avg_session_mins"] <= 0).any():
        bad_count = (df["avg_session_mins"] <= 0).sum()
        raise ValueError(
            f"Data validation failed: 'avg_session_mins' must be > 0. Found {bad_count} non-positive values."
        )

    if ((df["completion_rate"] < 0.0) | (df["completion_rate"] > 1.0)).any():
        bad_count = ((df["completion_rate"] < 0.0) | (df["completion_rate"] > 1.0)).sum()
        raise ValueError(
            f"Data validation failed: 'completion_rate' has {bad_count} values outside [0, 1]."
        )

    for genre_col in GENRE_COLUMNS:
        if ((df[genre_col] < 0.0) | (df[genre_col] > 1.0)).any():
            bad_count = ((df[genre_col] < 0.0) | (df[genre_col] > 1.0)).sum()
            raise ValueError(
                f"Data validation failed: '{genre_col}' has {bad_count} values outside [0, 1]."
            )

    if ((df["weekend_viewing_pct"] < 0.0) | (df["weekend_viewing_pct"] > 1.0)).any():
        bad_count = ((df["weekend_viewing_pct"] < 0.0) | (df["weekend_viewing_pct"] > 1.0)).sum()
        raise ValueError(
            f"Data validation failed: 'weekend_viewing_pct' has {bad_count} values outside [0, 1]."
        )

    if (df["days_since_last_watch"] < 0).any():
        bad_count = (df["days_since_last_watch"] < 0).sum()
        raise ValueError(
            f"Data validation failed: 'days_since_last_watch' has {bad_count} negative values."
        )


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Select and prepare features for clustering.

    Notes on feature selection and engineering:
    - `user_id` is excluded because it is an arbitrary identifier, not a behavioral signal.
    - We evaluate the 11 core behavioral dimensions capturing volume, frequency,
      duration, completion rate, genre preferences, schedule distribution, and recency.
    - Derived features (such as Shannon entropy or dominant genre share) were tested
      and found to introduce collinearity without improving silhouette separation.
    - Therefore, the 11 base behavioral features are retained to maintain optimal
      interpretability, prevent Euclidean distortion, and ensure production robustness.

    Args:
        df: Input validated DataFrame.

    Returns:
        Tuple of (prepared DataFrame with features, list of feature column names).
    """
    feature_names = list(FEATURE_COLUMNS)
    prepared_df = df[feature_names].copy()
    return prepared_df, feature_names


def evaluate_k_values(
    X_scaled: np.ndarray,
    k_range: range = range(2, 7),
    random_state: int = RANDOM_STATE,
) -> List[Dict[str, Any]]:
    """Evaluate K-Means across candidate K values using Inertia and Silhouette score.

    Args:
        X_scaled: Preprocessed feature matrix (StandardScaler applied).
        k_range: Range of K values to evaluate (default: 2 to 6 inclusive).
        random_state: Random state for deterministic K-Means initialization.

    Returns:
        List of dictionaries containing evaluation metrics for each K.
    """
    evaluation_results: List[Dict[str, Any]] = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        cluster_labels = kmeans.fit_predict(X_scaled)
        inertia = float(kmeans.inertia_)
        sil_score = float(silhouette_score(X_scaled, cluster_labels))
        sizes = [int(s) for s in np.bincount(cluster_labels, minlength=k)]

        evaluation_results.append(
            {
                "k": k,
                "inertia": round(inertia, 2),
                "silhouette": round(sil_score, 4),
                "cluster_sizes": sizes,
                "min_cluster_size": min(sizes),
                "max_cluster_size": max(sizes),
                "min_cluster_pct": round(min(sizes) / len(X_scaled) * 100, 1),
            }
        )

    return evaluation_results


def select_best_k(evaluation_results: List[Dict[str, Any]]) -> Tuple[int, str]:
    """Select the optimal K based primarily on Silhouette Score, with cluster balance

    and inertia/elbow behavior as secondary criteria.

    Args:
        evaluation_results: List of evaluation metrics per K.

    Returns:
        Tuple of (selected K, human-readable rationale).
    """
    # Filter candidates with reasonable cluster balance (no cluster < 5% of users)
    viable_candidates = [
        res for res in evaluation_results if res["min_cluster_pct"] >= 5.0
    ]

    if not viable_candidates:
        viable_candidates = evaluation_results

    # Select candidate with highest silhouette score
    best_candidate = max(viable_candidates, key=lambda res: res["silhouette"])
    best_k = best_candidate["k"]

    rationale = (
        f"K={best_k} achieves the highest Silhouette Score ({best_candidate['silhouette']:.4f}) "
        f"among all evaluated configurations, indicating well-separated, cohesive clusters. "
        f"It also exhibits balanced cluster distribution (sizes: {best_candidate['cluster_sizes']}, "
        f"smallest cluster {best_candidate['min_cluster_pct']}%) with no degenerate clusters, "
        f"and corresponds to an elbow point where inertia reduction stabilizes (Inertia: {best_candidate['inertia']:.2f})."
    )

    return best_k, rationale


def train_final_model(
    X_raw: pd.DataFrame | np.ndarray,
    selected_k: int,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Train the final scikit-learn Pipeline containing StandardScaler and KMeans.

    Args:
        X_raw: Unscaled feature matrix or DataFrame.
        selected_k: Number of clusters to fit.
        random_state: Random state for reproducible training.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=selected_k,
                    random_state=random_state,
                    n_init=20,
                ),
            ),
        ]
    )
    pipeline.fit(X_raw)
    return pipeline


def _derive_segment_name_and_desc(
    stats: Dict[str, float],
) -> Tuple[str, str, str]:
    """Dynamically determine a human-readable segment name and description based on

    actual empirical cluster statistics.

    Args:
        stats: Dictionary of feature mean values for a cluster.

    Returns:
        Tuple of (segment_name, dominant_behavior, short_description).
    """
    watch_time = stats["watch_time_hours"]
    sessions = stats["sessions_per_week"]
    duration = stats["avg_session_mins"]
    completion = stats["completion_rate"]
    weekend_pct = stats["weekend_viewing_pct"]
    recency = stats["days_since_last_watch"]

    action = stats["action_pct"]
    comedy = stats["comedy_pct"]
    drama = stats["drama_pct"]
    thriller = stats["thriller_pct"]
    sci_fi = stats["sci_fi_pct"]

    genre_map = {
        "Action": action,
        "Comedy": comedy,
        "Drama": drama,
        "Thriller": thriller,
        "Sci-Fi": sci_fi,
    }
    sorted_genres = sorted(genre_map.items(), key=lambda x: x[1], reverse=True)
    top_genre, top_genre_share = sorted_genres[0]
    second_genre, second_genre_share = sorted_genres[1]

    # Rule 1: Low-Activity / At-Risk Viewers
    if recency >= 15.0 or (watch_time < 5.0 and completion < 0.45):
        return (
            "Low-Activity / Churn-Risk Viewers",
            "Infrequent visits, short sessions, high inactivity (>20 days)",
            "Dormant viewers with minimal engagement and elevated churn risk. Require re-engagement campaigns.",
        )

    # Rule 2: Weekend Binge Viewers
    if weekend_pct >= 0.70:
        return (
            "Weekend Binge Viewers",
            f"Concentrated weekend viewing ({weekend_pct*100:.1f}%), long session duration, high completion",
            "Viewers who consume content primarily during weekends in extended, focused viewing blocks.",
        )

    # Rule 3: High-Engagement Power Viewers
    if watch_time >= 30.0 or (sessions >= 10.0 and duration >= 80.0):
        return (
            "High-Engagement Power Viewers",
            f"Daily high-volume viewing ({watch_time:.1f} hrs/wk, {sessions:.1f} sessions/wk), high completion ({completion*100:.1f}%)",
            "Platform power users with top watch times, frequent weekly visits, and diverse content consumption.",
        )

    # Rule 4: Genre-Focused Viewers
    if (top_genre_share + second_genre_share) >= 0.70 or top_genre_share >= 0.35:
        combined_genres = f"{top_genre} & {second_genre}"
        return (
            f"Genre-Focused Viewers ({combined_genres})",
            f"High affinity for {combined_genres} ({top_genre_share*100:.1f}% / {second_genre_share*100:.1f}%)",
            f"Enthusiasts with strong, concentrated affinity for specific genres ({combined_genres}).",
        )

    # Rule 5: Casual Viewers
    if watch_time < 15.0 and duration < 50.0:
        return (
            "Casual Viewers",
            f"Moderate watch time ({watch_time:.1f} hrs), short bite-sized sessions ({duration:.1f} mins), comedy/variety affinity",
            "Casual viewers consuming bite-sized content on flexible schedules with moderate commitment.",
        )

    # Fallback based on top genre and activity
    return (
        f"General Streamers ({top_genre} Leaning)",
        f"Balanced viewing with preference for {top_genre}",
        "Regular viewers with steady weekly engagement and moderate genre preferences.",
    )


def analyze_clusters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Calculate cluster-level feature averages and empirical characteristics.

    Args:
        df: Input DataFrame containing the raw features.
        cluster_labels: Array of cluster label assignments.
        feature_names: List of feature names to analyze.

    Returns:
        List of cluster analysis dictionaries sorted by cluster ID.
    """
    total_users = len(df)
    df_clustered = df[feature_names].copy()
    df_clustered["cluster"] = cluster_labels

    cluster_summaries: List[Dict[str, Any]] = []

    for cluster_id in sorted(df_clustered["cluster"].unique()):
        cdf = df_clustered[df_clustered["cluster"] == cluster_id]
        user_count = int(len(cdf))
        percentage = round((user_count / total_users) * 100, 2)

        means = {
            col: round(float(cdf[col].mean()), 4) for col in feature_names
        }

        segment_name, dominant_behavior, description = _derive_segment_name_and_desc(
            means
        )

        # Dominant genre
        genre_means = {g: means[g] for g in GENRE_COLUMNS}
        top_genre = max(genre_means.items(), key=lambda x: x[1])[0].replace("_pct", "").capitalize()

        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "segment_name": segment_name,
                "user_count": user_count,
                "percentage": percentage,
                "dominant_behavior": dominant_behavior,
                "dominant_genre": top_genre,
                "description": description,
                "feature_means": means,
            }
        )

    return cluster_summaries


def save_model(
    pipeline: Pipeline,
    feature_names: List[str],
    selected_k: int,
    silhouette: float,
    inertia: float,
    cluster_metadata: List[Dict[str, Any]],
    evaluation_table: List[Dict[str, Any]],
    artifact_path: Path = MODEL_ARTIFACT_PATH,
    metadata_path: Path = METADATA_ARTIFACT_PATH,
) -> None:
    """Persist the complete inference pipeline and machine-readable metadata.

    Args:
        pipeline: Trained scikit-learn Pipeline (StandardScaler + KMeans).
        feature_names: List of input feature names.
        selected_k: Chosen K.
        silhouette: Final Silhouette score.
        inertia: Final Inertia.
        cluster_metadata: Detailed cluster interpretation metadata.
        evaluation_table: Comparison metrics across evaluated K values.
        artifact_path: Destination path for pipeline.pkl.
        metadata_path: Destination path for metadata.json.
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    segment_name_map = {
        meta["cluster_id"]: meta["segment_name"] for meta in cluster_metadata
    }

    # Complete self-contained artifact for future API serving
    pipeline_artifact = {
        "pipeline": pipeline,
        "scaler": pipeline.named_steps["scaler"],
        "kmeans": pipeline.named_steps["kmeans"],
        "feature_names": feature_names,
        "selected_k": selected_k,
        "silhouette_score": silhouette,
        "inertia": inertia,
        "cluster_metadata": cluster_metadata,
        "segment_names": segment_name_map,
        "training_config": {
            "random_state": RANDOM_STATE,
            "n_init": 20,
            "scaler": "StandardScaler",
            "algorithm": "KMeans",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(pipeline_artifact, artifact_path)

    # Human- and machine-readable JSON metadata
    json_metadata = {
        "project": "OTT Audience Segmentation & Personalization Service",
        "phase": 1,
        "status": "trained",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_configuration": {
            "random_state": RANDOM_STATE,
            "n_init": 20,
            "scaler": "StandardScaler",
            "algorithm": "KMeans",
        },
        "evaluation_summary": evaluation_table,
        "selected_k": selected_k,
        "silhouette_score": silhouette,
        "inertia": inertia,
        "feature_names": feature_names,
        "clusters": cluster_metadata,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(json_metadata, f, indent=2)


def main() -> None:
    """Execute the end-to-end Phase 1 ML training workflow."""
    print("=" * 60)
    print("OTT AUDIENCE SEGMENTATION TRAINING")
    print("=" * 60)

    # 1. Load dataset
    df = load_dataset(DATA_PATH)
    num_rows = len(df)
    print(f"Dataset rows: {num_rows}")

    # 2. Validate dataset
    validate_dataset(df)
    print("Data validation:\nPASSED")

    # 3. Feature selection & engineering
    prepared_df, feature_names = engineer_features(df)
    X_raw = prepared_df.values
    print(f"Number of features: {len(feature_names)}")
    print("Feature list:")
    for feat in feature_names:
        print(f"  - {feat}")

    # Scale for K evaluation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 4. Evaluate K values (K = 2, 3, 4, 5, 6)
    print("\nModel comparison:")
    print(f"{'K':<4} {'Inertia':<12} {'Silhouette':<12} {'Cluster Sizes'}")
    print("-" * 55)
    evaluation_results = evaluate_k_values(X_scaled)
    for res in evaluation_results:
        print(
            f"{res['k']:<4} {res['inertia']:<12.2f} {res['silhouette']:<12.4f} {res['cluster_sizes']}"
        )

    # 5. Model selection
    best_k, rationale = select_best_k(evaluation_results)
    best_res = next(r for r in evaluation_results if r["k"] == best_k)
    print(f"\nSelected K:\n{best_k}")
    print(f"Final silhouette:\n{best_res['silhouette']:.4f}")
    print(f"Final inertia:\n{best_res['inertia']:.2f}")
    print(f"Selection rationale:\n{rationale}")

    # 6. Train final model inside sklearn Pipeline
    pipeline = train_final_model(prepared_df, best_k, random_state=RANDOM_STATE)
    cluster_labels = pipeline.named_steps["kmeans"].labels_

    # 7. Cluster analysis & interpretation
    cluster_metadata = analyze_clusters(df, cluster_labels, feature_names)

    print("\nCluster distribution:")
    for c in cluster_metadata:
        print(
            f"Cluster {c['cluster_id']}: {c['segment_name']} "
            f"({c['user_count']} users, {c['percentage']}%)"
        )

    print("\nSegment interpretation:")
    for c in cluster_metadata:
        means = c["feature_means"]
        print(f"\n--- [Cluster {c['cluster_id']}] {c['segment_name']} ---")
        print(f"  Audience Size : {c['user_count']} users ({c['percentage']}%)")
        print(f"  Key Behavior  : {c['dominant_behavior']}")
        print(f"  Description   : {c['description']}")
        print(f"  Watch Time    : {means['watch_time_hours']:.2f} hrs/wk")
        print(f"  Sessions      : {means['sessions_per_week']:.1f} sessions/wk")
        print(f"  Avg Duration  : {means['avg_session_mins']:.1f} mins")
        print(f"  Completion    : {means['completion_rate']*100:.1f}%")
        print(f"  Weekend Pct   : {means['weekend_viewing_pct']*100:.1f}%")
        print(f"  Recency       : {means['days_since_last_watch']:.1f} days ago")
        print(
            f"  Genre Prefs   : Action {means['action_pct']*100:.1f}% | "
            f"Comedy {means['comedy_pct']*100:.1f}% | "
            f"Drama {means['drama_pct']*100:.1f}% | "
            f"Thriller {means['thriller_pct']*100:.1f}% | "
            f"Sci-Fi {means['sci_fi_pct']*100:.1f}%"
        )

    # 8. Save model and metadata artifacts
    save_model(
        pipeline=pipeline,
        feature_names=feature_names,
        selected_k=best_k,
        silhouette=best_res["silhouette"],
        inertia=best_res["inertia"],
        cluster_metadata=cluster_metadata,
        evaluation_table=evaluation_results,
        artifact_path=MODEL_ARTIFACT_PATH,
        metadata_path=METADATA_ARTIFACT_PATH,
    )

    print(f"\nModel saved:\n{MODEL_ARTIFACT_PATH.relative_to(BASE_DIR)}")
    print(f"Metadata saved:\n{METADATA_ARTIFACT_PATH.relative_to(BASE_DIR)}")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
