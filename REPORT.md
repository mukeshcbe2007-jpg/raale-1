# Technical Report: Containerized Audience Segmentation & Personalization Service (Phase 1)

**Author:** Senior Machine Learning Engineer  
**Status:** Phase 1 Completed & Verified  
**Date:** October 2026  
**Repository:** `ott-audience-segmentation`  

---

## 1. Problem Statement

Modern Over-The-Top (OTT) streaming platforms experience high user churn, content discovery fatigue, and fragmented viewer attention. Viewers interact with platforms across distinct temporal, volume, and genre dimensions:

- Some binge high-budget drama series exclusively on weekends.
- Others stream 20-minute comedy sitcoms casually during weeknight dinners.
- A passionate cohort consumes only sci-fi and action blockbusters.
- Power users stream daily across diverse genres with high completion rates.
- Inactive subscribers visit rarely, leave titles unfinished, and steadily approach churn.

Manually tagging users with static demographic heuristics fails to capture the dynamic reality of viewer habits. Platforms require an automated, data-driven system to discover natural behavioral groupings without manual supervision.

---

## 2. Proposed Solution

We propose a multi-stage machine learning and microservice architecture:

$$\text{User Telemetry Data} \longrightarrow \text{Strict Validation} \longrightarrow \text{StandardScaler} \longrightarrow \text{K-Means Clustering} \longrightarrow \text{Dynamic Segment Interpretation} \longrightarrow \text{Personalization Engine}$$

In full production, this system is divided into three isolated, container-ready services:

1. **`trainer`**: Autonomous batch ML training pipeline that validates raw data, tunes clustering models, generates explainable segment profiles, and serializes production inference artifacts.
2. **`api`**: Fast, containerized REST service (FastAPI) loading the persisted pipeline to serve real-time user segment inference and personalized recommendations.
3. **`evaluator`**: Continuous monitoring service tracking silhouette decay, inertia drift, and cluster size stability over time.

---

## 3. Phase 1 Objective

Phase 1 focuses **exclusively on the core ML training system (`trainer`)**. The primary objectives are:

- Build a realistic, deterministic synthetic behavioral dataset (1,000 viewers, seed `42`).
- Enforce strict pre-flight data validation rules with descriptive failure exceptions.
- Formulate a leak-free scikit-learn preprocessing and clustering pipeline.
- Conduct empirical hyperparameter search across $K \in \{2, 3, 4, 5, 6\}$ using Silhouette Analysis and Inertia.
- Dynamically interpret discovered clusters into human-actionable audience segments.
- Persist self-contained artifacts (`models/pipeline.pkl` and `models/metadata.json`).
- Ensure 100% bit-for-bit reproducibility across independent runs.

---

## 4. Dataset

The dataset was generated deterministically and saved to `data/users.csv`. It comprises 1,000 unique viewer records and 12 columns.

### Feature Inventory and Statistics

| Column Name | Metric Type | Validation Constraint | Dataset Mean | Observed Range | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `user_id` | String Identifier | Unique non-null string | N/A | `USR_0001` - `USR_1000` | Excluded from ML |
| `watch_time_hours` | Continuous Float | $\ge 0.0$ hours | 17.76 hrs | 0.20 - 55.26 hrs | ML Feature |
| `sessions_per_week` | Continuous Float | $\ge 0.0$ sessions | 6.39 sessions | 0.30 - 19.10 sessions | ML Feature |
| `avg_session_mins` | Continuous Float | $> 0.0$ minutes | 61.61 mins | 12.00 - 138.10 mins | ML Feature |
| `completion_rate` | Ratio Float | $[0.0, 1.0]$ | 0.677 (67.7%) | 0.133 - 0.985 | ML Feature |
| `action_pct` | Ratio Float | $[0.0, 1.0]$ | 0.241 (24.1%) | 0.009 - 0.764 | ML Feature |
| `comedy_pct` | Ratio Float | $[0.0, 1.0]$ | 0.197 (19.7%) | 0.000 - 0.674 | ML Feature |
| `drama_pct` | Ratio Float | $[0.0, 1.0]$ | 0.178 (17.8%) | 0.000 - 0.599 | ML Feature |
| `thriller_pct` | Ratio Float | $[0.0, 1.0]$ | 0.172 (17.2%) | 0.000 - 0.712 | ML Feature |
| `sci_fi_pct` | Ratio Float | $[0.0, 1.0]$ | 0.212 (21.2%) | 0.002 - 0.724 | ML Feature |
| `weekend_viewing_pct` | Ratio Float | $[0.0, 1.0]$ | 0.476 (47.6%) | 0.120 - 0.941 | ML Feature |
| `days_since_last_watch` | Continuous Float | $\ge 0.0$ days | 6.58 days | 0.00 - 42.00 days | ML Feature |

> **Unsupervised Integrity**: The dataset contains zero label or cluster columns. The data generation process sampled five distinct behavioral distributions to provide natural density clusters, but user rows were randomly shuffled and unlabelled.

---

## 5. Feature Engineering Analysis

During pipeline design, we empirically evaluated whether deriving additional features—such as Shannon genre entropy ($H = -\sum p_i \log_2 p_i$) and dominant genre percentage ($\max p_i$)—improved clustering separation.

### Empirical Feature Comparison Experiment

| Configuration | Feature Count | Optimal K | Silhouette Score | Minimum Cluster Size | Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Base 11 Features** | **11** | **5** | **0.3523** | **16.6% (166 users)** | **Superior separation; natural alignment with behavioral cohorts.** |
| With Derived Features | 13 | 5 | 0.3269 | 16.8% (168 users) | Silhouette degraded by -0.0254; introduced collinearity. |

### Technical Decision

Adding derived summary statistics on top of the already complete genre distribution ($\sum p_{\text{genre}} = 1.0$) caused collinearity and Euclidean distance inflation (the "curse of dimensionality"). The 11 base behavioral features are orthogonal, easily interpretable, and yield higher cluster cohesion ($0.3523$ vs $0.3269$). Therefore, the **11 core features were selected** for production.

---

## 6. Data Validation

In `trainer/train.py`, `validate_dataset(df)` implements strict pre-flight gatekeeping. If any condition is violated, execution terminates immediately with a descriptive exception:

1. **Presence & Non-Emptiness**: Guarantees file existence and verified row count.
2. **Schema Integrity**: Verifies all 12 required columns exist in the incoming dataframe.
3. **Identifier Uniqueness**: Asserts `df['user_id'].duplicated().any() == False`.
4. **Zero Missingness**: Asserts zero `NaN`, `None`, or empty strings across all columns.
5. **Numeric Typing**: Validates all feature columns conform to numeric datatypes (`float64`, `int64`).
6. **Domain Boundary Constraints**:
   - `watch_time_hours >= 0`
   - `sessions_per_week >= 0`
   - `avg_session_mins > 0` (strictly positive session duration)
   - `completion_rate \in [0.0, 1.0]`
   - `action_pct, comedy_pct, drama_pct, thriller_pct, sci_fi_pct \in [0.0, 1.0]`
   - `weekend_viewing_pct \in [0.0, 1.0]`
   - `days_since_last_watch >= 0`

---

## 7. Preprocessing

Different behavioral features operate on drastically different physical scales:

- `watch_time_hours`: $0 - 55$
- `avg_session_mins`: $12 - 138$
- `days_since_last_watch`: $0 - 42$
- `completion_rate`, genre percentages, `weekend_viewing_pct`: $0.0 - 1.0$

Because K-Means utilizes Euclidean distance, unscaled features with large absolute ranges would completely dominate distance calculations, rendering genre preferences and completion rates mathematically irrelevant.

We utilize `StandardScaler` from scikit-learn:
$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

To prevent data leakage and guarantee that future REST API requests undergo the exact same mathematical transformation, `StandardScaler` is encapsulated with the clustering model inside a reusable `sklearn.pipeline.Pipeline`.

---

## 8. Machine Learning Model

We chose **K-Means Clustering** as the primary segmentation algorithm for Phase 1:

- **Computational Efficiency**: Fits in milliseconds and performs sub-millisecond inference during real-time API queries.
- **Explainability**: Centroids directly map to real-world viewer behavior averages that can be communicated to business executives and marketing teams.
- **Containerization Readiness**: No GPU or heavy framework dependencies (e.g. PyTorch/TensorFlow); keeps Docker images small and lightweight (<250MB).
- **Robustness**: Initialized with `n_init=20` and `random_state=42` to eliminate local optima instability.

---

## 9. Model Selection

We evaluated candidate $K \in \{2, 3, 4, 5, 6\}$ on the preprocessed feature matrix.

### Model Evaluation Table (Actual Execution Metrics)

| K | Inertia | Silhouette Score | Cluster Size Distribution | Smallest Cluster % | Qualitative Assessment |
| :-: | :-----: | :--------------: | :-----------------------: | :----------------: | :--------------------- |
| **2** | 7464.22 | 0.3016 | [591, 409] | 40.9% | Coarse high vs. low activity split; loses all genre and schedule nuances. |
| **3** | 5969.50 | 0.2896 | [206, 393, 401] | 20.6% | Poor separation; merges weekend viewers and power streamers. |
| **4** | 4798.36 | 0.3348 | [217, 192, 399, 192] | 19.2% | Promising, but merges casual viewers and low-activity viewers into one block (n=399). |
| **5** | **3861.80** | **0.3523** | **[166, 191, 238, 217, 188]** | **16.6%** | **Optimal. Peak silhouette score, balanced cluster sizes, clear behavioral boundaries.** |
| **6** | 3628.36 | 0.3091 | [183, 191, 96, 163, 216, 151] | 9.6% | Over-fragmentation; generates an undersized cluster (n=96) and drops silhouette. |

### Selection Rationale

1. **Primary Metric (Silhouette Score)**: $K=5$ achieves the global maximum silhouette score ($0.3523$), demonstrating that points within each cluster are tightly clustered relative to neighboring clusters.
2. **Secondary Metric (Inertia Elbow)**: Inertia drops significantly from $K=2$ ($7464.22$) to $K=5$ ($3861.80$). Beyond $K=5$, the reduction flattens ($K=6$ drops only to $3628.36$, an inertia reduction of only $6.0\%$).
3. **Cluster Balance**: The cluster distribution at $K=5$ is remarkably even ($16.6\%$, $19.1\%$, $23.8\%$, $21.7\%$, $18.8\%$). No cluster dominates or collapses into an outlier pocket.
4. **Actionable Segmentation**: Exactly mirrors 5 operational archetypes required by streaming marketing and editorial teams.

---

## 10. Clustering Evaluation

### Silhouette Analysis

- The final silhouette score is **`+0.3523`**.
- In customer behavioral segmentation involving continuous high-dimensional behavioral signals, a silhouette between $0.30$ and $0.40$ reflects strong, naturally overlapping human behavioral boundaries without artificial clustering collapse.

### Inertia Analysis

- Final inertia is **`3861.80`**.
- While higher $K$ values always decrease inertia mechanically ($K=N \implies \text{Inertia}=0$), selecting $K$ solely on inertia is an anti-pattern. The elbow test confirms $K=5$ balances cluster tightness without overfitting.

### Cluster Balance Check

- Largest cluster: 238 users (23.8%)
- Smallest cluster: 166 users (16.6%)
- Ratio of largest to smallest: $1.43:1$
- Result: **Zero degenerate clusters** detected.

---

## 11. Cluster Interpretation & Discovered Segments

Cluster IDs assigned by K-Means ($0, 1, 2, 3, 4$) are arbitrary mathematical identifiers. In `trainer/train.py`, we implemented a heuristic interpretation engine `analyze_clusters` that analyzes the empirical centroid of each cluster to determine its human-readable identity.

### Empirical Cluster Profiles ($K=5$)

#### Cluster 0: Low-Activity / Churn-Risk Viewers

- **Audience Size**: 166 viewers (16.60% of total)
- **Key Characteristics**:
  - Watch Time: $1.84$ hours/week (lowest across platform)
  - Frequency: $1.21$ sessions/week
  - Avg Session Duration: $23.83$ minutes
  - Completion Rate: $31.50\%$ (majority of videos abandoned)
  - Viewing Recency: **$25.47$ days since last watch** (critical inactivity indicator)
  - Genre Distribution: Diffuse (Action 18.6%, Comedy 20.0%, Drama 18.9%, Thriller 22.7%, Sci-Fi 19.9%)
- **Strategic Personalization Action**: Immediate automated win-back campaigns; push notifications highlighting newly released sequels of past-watched titles; email digest of trending content.

#### Cluster 1: Weekend Binge Viewers

- **Audience Size**: 191 viewers (19.10% of total)
- **Key Characteristics**:
  - Watch Time: $18.74$ hours/week
  - Frequency: $4.48$ sessions/week
  - Avg Session Duration: $82.40$ minutes (long, immersive sessions)
  - Completion Rate: $81.64\%$
  - Weekend Concentration: **$82.23\%$ of all streaming occurs on weekends**
  - Viewing Recency: $2.70$ days ago
  - Genre Distribution: Leans heavily toward Drama ($27.27\%$) and Thriller ($26.67\%$)
- **Strategic Personalization Action**: Schedule personalized push notifications for Friday 6:00 PM; surface multi-episode limited series and cinematic crime thrillers on homepage hero carousels.

#### Cluster 2: Casual Viewers

- **Audience Size**: 238 viewers (23.80% of total)
- **Key Characteristics**:
  - Watch Time: $8.36$ hours/week
  - Frequency: $3.72$ sessions/week
  - Avg Session Duration: $36.28$ minutes (short, bite-sized sessions)
  - Completion Rate: $57.14\%$
  - Weekend Concentration: $38.21\%$ (balanced weekday viewing)
  - Viewing Recency: $4.79$ days ago
  - Genre Distribution: **Dominant preference for Comedy ($33.44\%$)** and Action ($24.15\%$)
- **Strategic Personalization Action**: Promote 20-30 minute comedy series, stand-up comedy specials, and digestible serialized episodic content on the homepage row.

#### Cluster 3: High-Engagement Power Viewers

- **Audience Size**: 217 viewers (21.70% of total)
- **Key Characteristics**:
  - Watch Time: **$36.01$ hours/week** (highest on platform)
  - Frequency: **$14.15$ sessions/week** (multiple sessions daily)
  - Avg Session Duration: $94.26$ minutes
  - Completion Rate: **$88.09\%$**
  - Weekend Concentration: $42.39\%$ (steady streaming every single day)
  - Viewing Recency: **$1.31$ days ago** (active platform residents)
  - Genre Distribution: Broad catalog exploration (Action 20.7%, Comedy 19.9%, Drama 19.8%, Thriller 20.4%, Sci-Fi 19.2%)
- **Strategic Personalization Action**: Early access to platform originals; invitation to beta features; premium 4K Ultra-HD / Dolby Atmos content recommendations; VIP loyalty perks.

#### Cluster 4: Genre-Focused Viewers (Action & Sci-Fi)

- **Audience Size**: 188 viewers (18.80% of total)
- **Key Characteristics**:
  - Watch Time: $21.68$ hours/week
  - Frequency: $7.35$ sessions/week
  - Avg Session Duration: $68.24$ minutes
  - Completion Rate: $75.61\%$
  - Weekend Concentration: $43.42\%$
  - Viewing Recency: $2.22$ days ago
  - Genre Distribution: **Action ($41.00\%$) and Sci-Fi ($40.59\%$) comprise $81.59\%$ of all viewing** (Comedy: 6.6%, Drama: 6.5%, Thriller: 5.3%)
- **Strategic Personalization Action**: Curate home feeds dedicated to sci-fi franchises, superhero universes, anime, and cyberpunk action releases; exclude romantic drama suggestions.

---

## 12. Reproducibility & Stability Verification

To prove production reliability, the training pipeline was executed sequentially in independent processes.

### Verification Run Comparison

| Metric / Attribute | Run 1 | Run 2 | Status |
| :--- | :--- | :--- | :--- |
| Random Seed Configuration | `random_state=42, n_init=20` | `random_state=42, n_init=20` | Deterministic |
| Dataset Row Count | 1,000 | 1,000 | Identical |
| Selected K | 5 | 5 | Identical |
| Final Silhouette Score | **`0.3523`** | **`0.3523`** | Bit-for-bit identical |
| Final Inertia | **`3861.80`** | **`3861.80`** | Bit-for-bit identical |
| Cluster Distribution | `[166, 191, 238, 217, 188]` | `[166, 191, 238, 217, 188]` | Bit-for-bit identical |
| Cluster Centroid Values | Identical to 4 decimal places | Identical to 4 decimal places | Verified |

Result: **The training pipeline is fully reproducible.**

---

## 13. Model Artifacts

Two complementary production artifacts are generated and persisted into `models/`:

### 1. `models/pipeline.pkl` (Binary Inference Artifact)

- Serialized using `joblib`.
- Structure:

  ```python
  {
      "pipeline": Pipeline(steps=[("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=5))]),
      "scaler": StandardScaler(),
      "kmeans": KMeans(n_clusters=5),
      "feature_names": [...],
      "selected_k": 5,
      "silhouette_score": 0.3523,
      "inertia": 3861.80,
      "cluster_metadata": [...],
      "segment_names": {
          0: "Low-Activity / Churn-Risk Viewers",
          1: "Weekend Binge Viewers",
          2: "Casual Viewers",
          3: "High-Engagement Power Viewers",
          4: "Genre-Focused Viewers (Action & Sci-Fi)",
      },
      "training_config": {"random_state": 42, "n_init": 20, "scaler": "StandardScaler"},
      "created_at": "2026-10-05T13:01:46Z",
  }
  ```

- **Inference Verification**: Loaded into an independent Python runtime and validated on unseen user records:

  ```python
  import joblib

  art = joblib.load("models/pipeline.pkl")
  pred_cluster = art["pipeline"].predict(sample_features)[0]
  segment = art["segment_names"][pred_cluster]
  ```

### 2. `models/metadata.json` (Machine-Readable Contract)

- JSON document containing full training metadata, hyperparameters, silhouette comparisons across all $K$, and full centroid summaries. This allows the future evaluator microservice to read metrics without unpickling Python objects.

---

## 14. Limitations

1. **Synthetic Telemetry Baseline**: While modeled after real-world OTT consumption distributions (Dirichlet genre vectors, clipped Gaussians for watch times), real-world streaming data contains seasonality (holidays, summer vacations) and content catalog release shocks.
2. **K-Means Spherical Boundary Assumption**: K-Means assumes isotropic spherical clusters. In future iterations, density-based algorithms (such as HDBSCAN) or Gaussian Mixture Models (GMM) can be evaluated to capture non-spherical clusters.
3. **Static Segment Assignment**: Real viewers transition across segments over time (e.g. a power viewer becoming casual). A hidden Markov or temporal graph model would be required for temporal state transitions.

---

## 15. Next Phase & Future Architecture

### Upcoming Microservices

1. **API Service (`api`)**:
   - Built with FastAPI.
   - Endpoints:
     - `POST /v1/segment/predict`: Accepts real-time user behavior payload, returns predicted cluster, human-readable segment name, and personalized recommendations.
     - `GET /v1/segments`: Returns all active segment profiles and centroid stats.
     - `GET /v1/health`: Liveness probe.
2. **Evaluator Service (`evaluator`)**:
   - Automated drift detector comparing incoming daily telemetry against `models/metadata.json`.
   - Alerts if silhouette drops below $0.25$ or cluster balance exceeds $60\%$.
3. **Containerization**:
   - Multi-stage Dockerfiles for `trainer`, `api`, and `evaluator`.
   - `docker-compose.yml` for unified local and staging orchestration.

---

### Future OTT UI/UX Specification (Preserved for Frontend Phase)

When the frontend demo dashboard is built, it will strictly adopt the following design language:

#### Visual Identity: "AI-Powered OTT Audience Intelligence"

Combines the immersive cinematic quality of premium movie streaming services with the clarity of modern AI analytics dashboards.

#### Cinematic Color Palette

- **Deep Background**: `#0B0B0F` (Night Sky Dark)
- **Secondary Surface**: `#14141A` (Charcoal Tint)
- **Card Background**: `#1C1C24` (Elevated Panel)
- **Primary Accent**: `#E50914` (Cinematic Streaming Red)
- **Accent Hover**: `#FF2A35` (Vibrant Crimson)
- **Primary Text**: `#FFFFFF` (Pure White)
- **Secondary Text**: `#A1A1AA` (Muted Zinc)
- **Borders & Dividers**: `#2A2A32`
- **Success / Warning Indicators**: `#22C55E` (Emerald) / `#F59E0B` (Amber)

#### Typography & Design Elements

- Font: **Plus Jakarta Sans** or **Inter**.
- Clean, non-distracting UI with film-strip inspired card dividers and subtle glow highlights.
- Metric Cards: Total Viewers (1,000), Active Segments (5), Optimal Silhouette Score (0.3523), Churn Risk Cohort (16.6%).
- Interactive Segment Cards displaying verified segment names, user counts, percentages, and dominant behaviors directly consumed from `models/metadata.json` (no hard-coded or fake statistics).

---

## 16. Phase 2: REST API Architecture & Inference Engineering

### 16.1 Architecture Overview

Phase 2 transitions the persisted ML model into an operational microservice using **FastAPI**:

```text
[ Client Request ]
        │
        ▼
[ Pydantic Schema Validation ]  (HTTP 422 if invalid constraints)
        │
        ▼
[ Controlled Feature Ordering ] (Exact Phase 1 sequence match)
        │
        ▼
[ Persisted scikit-learn Pipeline ] (StandardScaler -> KMeans)
        │
        ▼
[ K-Means Cluster Prediction ]  (Cluster ID assignment, read-only)
        │
        ▼
[ Distance-to-Centroid Computation ] (Euclidean metric in scaled space)
        │
        ▼
[ Segment Metadata Lookup ]     (Resolves human-readable identity)
        │
        ▼
[ Rule-Based Personalization ]  (Curated catalog matching)
        │
        ▼
[ Clean JSON Response ]         (HTTP 200 OK)
```

### 16.2 API Contract

#### `GET /health`

- **Purpose**: Liveness and model readiness probe.
- **Successful Response (HTTP 200)**:

  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

- **Degraded Response (HTTP 503)**:

  ```json
  {
    "status": "not_ready",
    "model_loaded": false
  }
  ```

#### `POST /recommend`

- **Purpose**: Real-time audience segment prediction and personalized content delivery.
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:

  ```json
  {
    "user_id": "USR-8192",
    "watch_time_hours": 32.5,
    "sessions_per_week": 5.0,
    "avg_session_mins": 85.0,
    "completion_rate": 0.82,
    "action_pct": 0.45,
    "comedy_pct": 0.10,
    "drama_pct": 0.15,
    "thriller_pct": 0.20,
    "sci_fi_pct": 0.10,
    "weekend_viewing_pct": 0.65,
    "days_since_last_watch": 2.0
  }
  ```

- **Response (HTTP 200 OK)**:

  ```json
  {
    "user_id": "USR-8192",
    "segment_id": 1,
    "segment_name": "Weekend Binge Viewers",
    "recommendations": [
      "Weekend Escape",
      "The Silent Harbor",
      "City Stories"
    ],
    "distance_to_centroid": 2.7917
  }
  ```

### 16.3 Input Validation Rules

All incoming payloads are strictly validated before any ML computation occurs:

- `user_id`: Non-empty, non-whitespace string.
- `watch_time_hours`, `sessions_per_week`, `days_since_last_watch`: $\ge 0.0$.
- `avg_session_mins`: $> 0.0$ (strictly positive session duration).
- `completion_rate`, genre percentages, `weekend_viewing_pct`: $[0.0, 1.0]$.
- Violations immediately return clean HTTP 422 JSON detailing the exact field without disclosing server paths or stack traces.

### 16.4 Inference Design Principles

1. **Single Startup Loading**: Model loading occurs exclusively within FastAPI's `lifespan` context manager. Objects (`pipeline`, `scaler`, `kmeans`, `metadata`) are cached in `app.state`.
2. **Zero Retraining Guarantee**: The model is never fitted or modified during API queries. The KMeans cluster centers and scaler parameters remain strictly immutable.
3. **Distance to Centroid**: Computes $\|\mathbf{x}_{\text{scaled}} - \mathbf{\mu}_c\|_2$. This provides an explainable confidence proxy indicating geometric alignment with the segment archetype.

### 16.5 Phase 2 Limitations

- **Transparent Heuristic Recommendation Layer**: The current recommendation output maps segments to curated lists from a static catalog (`CATALOG`) with genre-based fine-tuning. This is a lightweight personalization demonstration and not a full matrix-factorization / two-tower recommendation system.

---

## 17. Phase 3: Independent Evaluation & Quality Assurance

### 17.1 Evaluation Strategy

Automated, decoupled evaluation is essential for production ML pipelines to guarantee that:

1. The REST API behaves strictly as an inference-only microservice without unintentional retraining or state mutation.
2. The ML model’s internal clustering geometry remains completely intact during serving.
3. Edge cases and malformed inputs are cleanly rejected at the boundary without crashing the server process.

The evaluator service (`evaluator/evaluate.py`) executes as an independent HTTP client, decoupled from FastAPI internal application modules.

### 17.2 Clustering Quality Verification

The evaluator directly verified the empirical metrics persisted in `models/metadata.json`:

- **Selected Number of Clusters ($K$)**: **`5`**
- **Optimal Silhouette Score**: **`0.3523`**
- **Inertia at $K=5$**: **`3861.80`**
- **Cluster Balance Profile**:
  - Smallest Cluster (Cluster 0: Low-Activity): $166$ users ($16.60\%$)
  - Largest Cluster (Cluster 2: Casual): $238$ users ($23.80\%$)
  - Balance Ratio (Largest : Smallest): **$1.43 : 1$** (Healthy, non-degenerate distribution)

### 17.3 API Protocol & Schema Validation

All representative viewer personas were submitted to `POST /recommend`:

- **API Health**: Passed (`GET /health` returned `200 OK`, `model_loaded: true`).
- **Valid Requests**: **$5/5$ Passed** ($100.0\%$ success rate).
- **Schema Conformity**: Verified that every response contains `user_id`, `segment_id`, `segment_name`, `recommendations` (non-empty list), and `distance_to_centroid` ($\ge 0.0$).
- **Centroid Distances**: Verified all returned distances are valid non-negative Euclidean metrics.

### 17.4 Negative Input & Edge Case Testing

1. **Invalid Inputs ($6/6$ Passed)**:
   - Missing required field: Correctly returned `422 Unprocessable Content`.
   - Non-numeric string: Correctly returned `422 Unprocessable Content`.
   - Negative watch time: Correctly returned `422 Unprocessable Content`.
   - Completion rate $> 1.0$: Correctly returned `422 Unprocessable Content`.
   - Negative genre ratio: Correctly returned `422 Unprocessable Content`.
   - Blank / whitespace user ID: Correctly returned `422 Unprocessable Content`.
2. **Edge Cases ($4/4$ Passed)**:
   - Zero activity (`watch_time_hours=0.0`, `sessions_per_week=0.0`): Handled gracefully with `200 OK`.
   - Minimal activity (very small non-zero floats): Handled gracefully with `200 OK`.
   - Maximum activity (power user boundary values): Handled gracefully with `200 OK`.
   - 100% single-genre preference: Handled gracefully with `200 OK`.

### 17.5 Determinism & No-Retraining Proof

The evaluator transmitted the identical payload across 3 sequential iterations:

- Iteration 1: Segment ID `1` (`Weekend Binge Viewers`), Distance `2.7917`
- Iteration 2: Segment ID `1` (`Weekend Binge Viewers`), Distance `2.7917`
- Iteration 3: Segment ID `1` (`Weekend Binge Viewers`), Distance `2.7917`
- **Result**: $\Delta \text{distance} = 0.0000$. Centroids and assignments remained bit-for-bit constant, confirming read-only inference without model mutation.

### 17.6 Evaluation Evidence (`metrics.json`)

The complete audit metrics generated by `evaluator/evaluate.py` are persisted in [metrics.json](file:///c:/Users/Mukesh/OneDrive/Documents/Desktop/brahmastra/metrics.json):

```json
{
  "model": {
    "selected_k": 5,
    "silhouette_score": 0.3523,
    "inertia": 3861.8
  },
  "cluster_sizes": {
    "0": 166,
    "1": 191,
    "2": 238,
    "3": 217,
    "4": 188
  },
  "cluster_balance": {
    "smallest_cluster_id": 0,
    "smallest_cluster_size": 166,
    "smallest_cluster_pct": 16.6,
    "largest_cluster_id": 2,
    "largest_cluster_size": 238,
    "largest_cluster_pct": 23.8,
    "ratio_largest_to_smallest": 1.43,
    "observation": "Balanced distribution across all 5 segments with no degenerate clusters. Smallest cluster has 166 users (16.6%), largest has 238 users (23.8%)."
  },
  "api": {
    "api_url": "http://localhost:8000",
    "health": true,
    "total_requests": 18,
    "successful_requests": 18,
    "failed_requests": 0,
    "success_rate": 1.0
  },
  "validation": {
    "valid_profiles_passed": 5,
    "valid_profiles_total": 5,
    "invalid_input_handled": true,
    "invalid_tests_passed": 6,
    "invalid_tests_total": 6,
    "edge_cases_handled": true,
    "edge_cases_passed": 4,
    "edge_cases_total": 4,
    "prediction_consistent": true
  },
  "reproducibility": {
    "deterministic_prediction": true,
    "repeat_iterations_tested": 3,
    "no_retraining_verified": true
  }
}
```
