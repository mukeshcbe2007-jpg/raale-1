# Containerized Audience Segmentation & Personalization Service (Phase 1: Trainer)

An enterprise-ready, reproducible Machine Learning training pipeline designed for OTT streaming platforms to discover meaningful audience segments from behavioral viewer telemetry using unsupervised machine learning.

---

## 1. Problem

OTT streaming services (such as Netflix, Prime Video, Disney+, Hotstar) collect massive amounts of telemetry from daily viewer interactions:

- Total watch time and consumption velocity
- Session duration and frequency of visits
- Content completion and drop-off rates
- Granular genre affinities
- Day-of-week scheduling tendencies (weekday vs. weekend consumption)
- Inactivity intervals and viewing recency

Without automated segmentation, content catalog managers and marketing systems are forced to rely on coarse manual heuristics or static demographic stereotypes. To deliver hyper-personalized content recommendations, intelligent push notifications, UI homepage customization, and churn prevention campaigns, OTT platforms need to uncover latent behavioral cohorts directly from data—**without requiring manual or pre-assigned labels**.

---

## 2. Solution Architecture

The system discovers natural behavioral cohorts using an end-to-end unsupervised learning pipeline:

```text
+-------------------------------------------------------------+
|                     Raw Viewer Telemetry                    |
|  (watch time, session duration, genre split, recency, etc.) |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|             Data Validation & Feature Selection             |
|   (Strict domain boundaries, type checks, user_id excluded) |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|             StandardScaler Feature Normalization            |
|   (Zero-mean, unit-variance scaling for distance metrics)   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                 K-Means Clustering Pipeline                 |
|       (Candidate K=2..6 evaluated; K=5 optimal selected)    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|           Empirical Cluster Interpretation Engine           |
| (Dynamic rule-based naming derived from actual centroids)   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              Production Inference Artifacts                 |
|      (models/pipeline.pkl and models/metadata.json)         |
+-------------------------------------------------------------+
```

---

## 3. Phase 1 Scope

This repository represents **Phase 1** of the architecture: the **Production ML Trainer**.

- **Included in Phase 1**:
  - Deterministic synthetic OTT behavioral telemetry (`data/users.csv`, 1,000 users).
  - Strict pre-training data validation with actionable error reporting.
  - Standardized feature scaling and leak-free scikit-learn pipeline construction.
  - Multi-K evaluation ($K \in [2, 6]$) using Inertia, Silhouette Analysis, and Cluster Balance.
  - Data-driven segment interpretation and human-readable segment naming.
  - Model serialization (`models/pipeline.pkl`) and export of machine-readable metadata (`models/metadata.json`).
  - Bit-for-bit reproducible execution with fixed seeds.

- **Deferred to Future Phases**:
  - REST API serving service (`api`).
  - Automated continuous cluster evaluator service (`evaluator`).
  - Containerization and orchestration (`Dockerfile`, `docker-compose.yml`).
  - Cinematic web dashboard frontend.

---

## 4. Dataset

The dataset is located at `data/users.csv` and contains synthetic behavioral telemetry for 1,000 distinct OTT streaming viewers, generated with fixed random seed `42`.

### Behavioral Feature Definitions

| Feature Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `user_id` | String | `USR_0001` - `USR_1000` | Unique viewer identifier (**excluded from ML features**). |
| `watch_time_hours` | Float | $\ge 0.0$ hours | Total cumulative watch time recorded per week. |
| `sessions_per_week` | Float | $\ge 0.0$ sessions | Frequency of platform launch events per week. |
| `avg_session_mins` | Float | $> 0.0$ minutes | Mean length of individual streaming sessions. |
| `completion_rate` | Float | $[0.0, 1.0]$ | Proportion of started movies/episodes watched to completion. |
| `action_pct` | Float | $[0.0, 1.0]$ | Percentage of total watch time devoted to Action content. |
| `comedy_pct` | Float | $[0.0, 1.0]$ | Percentage of total watch time devoted to Comedy content. |
| `drama_pct` | Float | $[0.0, 1.0]$ | Percentage of total watch time devoted to Drama content. |
| `thriller_pct` | Float | $[0.0, 1.0]$ | Percentage of total watch time devoted to Thriller content. |
| `sci_fi_pct` | Float | $[0.0, 1.0]$ | Percentage of total watch time devoted to Sci-Fi content. |
| `weekend_viewing_pct` | Float | $[0.0, 1.0]$ | Fraction of streaming activity occurring on Saturday & Sunday. |
| `days_since_last_watch` | Float | $\ge 0.0$ days | Inactivity interval since viewer's last playback event. |

> **Unsupervised Integrity**: The dataset contains **zero** target labels, class tags, or manual segment annotations. All groupings are discovered purely through mathematical cluster cohesion.

---

## 5. Machine Learning Approach

### Why Exclude `user_id`?

`user_id` is an arbitrary identifier created by identity databases. Including user identifiers in distance-based clustering algorithms causes models to overfit to arbitrary database sequences rather than learning generalizable viewer behaviors.

### Feature Scaling: `StandardScaler`

K-Means calculates distances using Euclidean metrics:

$$d(\mathbf{x}, \mathbf{\mu}) = \sqrt{\sum_{j=1}^{D} (x_j - \mu_j)^2}$$

Features like `avg_session_mins` (scale: 10 - 150) would artificially dominate features like `completion_rate` (scale: 0.0 - 1.0) by several orders of magnitude. `StandardScaler` normalizes each feature to have a mean of 0 and standard deviation of 1 ($\mu = 0, \sigma = 1$), ensuring all behavioral dimensions contribute equitably.

### Algorithm: K-Means Clustering

K-Means is chosen as the foundational clustering algorithm because it is:

1. **Computationally Lightweight**: Rapid training and real-time sub-millisecond inference suitable for microservices.
2. **Transparent & Explainable**: Cluster centroids represent clear geometric averages of viewer behavior.
3. **Container-Ready**: Minimal memory footprint and rapid startup times inside Docker containers.

### Model Selection Criteria

Candidate models were evaluated across $K \in \{2, 3, 4, 5, 6\}$ using:

1. **Silhouette Score (Primary)**: Measures cohesion within clusters compared to separation from neighboring clusters (range: $[-1, +1]$).
2. **Inertia / Elbow Method (Secondary)**: Sum of squared distances to closest centroid; used to identify diminishing returns.
3. **Cluster Balance (Secondary)**: Rejects configurations with degenerate or fragmented clusters (<5% of population).
4. **Business Interpretability**: Ensures segments represent distinct, actionable marketing/recommendation targets.

---

## 6. How to Run

### Prerequisites

- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### Installation

From the project root:

```bash
pip install -r trainer/requirements.txt
```

### Run the Training Pipeline

```bash
python trainer/train.py
```

---

## 7. Actual Execution Results

Running `python trainer/train.py` yields the following verified output:

### Model Comparison Table

| K | Inertia | Silhouette Score | Cluster Size Distribution | Evaluation Summary |
| :-: | :-----: | :--------------: | :-----------------------: | :----------------- |
| **2** | 7464.22 | 0.3016 | [591, 409] | Coarse split into active vs. inactive users; lacks granularity. |
| **3** | 5969.50 | 0.2896 | [206, 393, 401] | Low silhouette; merges weekend binge and genre enthusiasts. |
| **4** | 4798.36 | 0.3348 | [217, 192, 399, 192] | Moderate silhouette; casual and power viewers partially merged. |
| **5** | **3861.80** | **0.3523** | **[166, 191, 238, 217, 188]** | **Optimal. Highest silhouette, balanced sizes, distinct profiles.** |
| **6** | 3628.36 | 0.3091 | [183, 191, 96, 163, 216, 151] | Over-fragmentation; creates an underpopulated cluster (n=96). |

### Selected Configuration: **$K = 5$**

- **Optimal Silhouette Score**: `0.3523`
- **Inertia**: `3861.80`

### Discovered Audience Segments ($K=5$)

```text
+-------------------------------------------------------------------------------------------------------+
| Cluster 0: Low-Activity / Churn-Risk Viewers (166 users, 16.6%)                                       |
| - Profile: 1.8 hrs/wk watch time | 1.2 sessions/wk | 23.8 min sessions | 25.5 days inactive           |
| - Action: Trigger automated re-engagement emails, win-back discounts, and trending digests.          |
+-------------------------------------------------------------------------------------------------------+
| Cluster 1: Weekend Binge Viewers (191 users, 19.1%)                                                   |
| - Profile: 18.7 hrs/wk watch time | 82.2% weekend activity | 82.4 min sessions | 81.6% completion     |
| - Action: Deliver Friday evening notifications for multi-part series, miniseries, and movie marathons.|
+-------------------------------------------------------------------------------------------------------+
| Cluster 2: Casual Viewers (238 users, 23.8%)                                                          |
| - Profile: 8.4 hrs/wk watch time | 3.7 sessions/wk | 36.3 min sessions | Comedy affinity (33.4%)      |
| - Action: Recommend 20-30 min sitcoms, stand-up specials, and bite-sized serialized episodes.         |
+-------------------------------------------------------------------------------------------------------+
| Cluster 3: High-Engagement Power Viewers (217 users, 21.7%)                                           |
| - Profile: 36.0 hrs/wk watch time | 14.2 sessions/wk | 94.3 min sessions | 88.1% completion           |
| - Action: Early access premieres, 4K HDR streams, VIP loyalty perks, and deep catalog recommendations.|
+-------------------------------------------------------------------------------------------------------+
| Cluster 4: Genre-Focused Viewers (Action & Sci-Fi) (188 users, 18.8%)                                 |
| - Profile: 21.7 hrs/wk watch time | 7.3 sessions/wk | 81.6% combined Action + Sci-Fi consumption      |
| - Action: Hero carousel customization with sci-fi blockbusters, franchises, and action thrillers.     |
+-------------------------------------------------------------------------------------------------------+
```

---

## 8. Persisted Artifacts

Upon completion, `trainer/train.py` creates two production-ready artifacts in `models/`:

1. **`models/pipeline.pkl`**:
   - Serialized via `joblib`.
   - Bundles the fitted `StandardScaler`, trained `KMeans` estimator, explicit `feature_names`, `cluster_metadata`, and `segment_names` dictionary.
   - Allows future microservices (FastAPI REST service) to perform single-line inference:

     ```python
     import joblib

     artifact = joblib.load("models/pipeline.pkl")
     pipeline = artifact["pipeline"]
     cluster_id = pipeline.predict(new_user_vector)[0]
     segment_name = artifact["segment_names"][cluster_id]
     ```

2. **`models/metadata.json`**:
   - Human- and machine-readable JSON document containing training timestamps, hyperparameters, K-evaluation summary, silhouette score, inertia, and full centroid profiles for automated service consumption.

---

## 9. Future OTT UI/UX Specification

When the dashboard frontend is built in later phases, it will implement the following design language:

- **Theme**: Cinematic Dark OTT Aesthetic.
- **Palette**:
  - Background: `#0B0B0F`
  - Secondary Surface: `#14141A`
  - Card Background: `#1C1C24`
  - Primary Accent: `#E50914` (Netflix Red)
  - Text: `#FFFFFF` (Primary), `#A1A1AA` (Secondary)
  - Success / Warning: `#22C55E` / `#F59E0B`
- **Typography**: Plus Jakarta Sans / Inter.
- **Visual Elements**: Subtle movie-poster cards, clean distribution charts, no fake numbers — all views bound directly to actual backend outputs from `models/metadata.json`.

---

## 10. Phase 2 — REST API Service

The Phase 2 microservice exposes the persisted Phase 1 clustering model as a high-performance, container-ready REST API built with **FastAPI** and **Pydantic**.

### Core Architecture & Guarantees

- **Zero Retraining During Inference**: The API strictly operates in read-only inference mode. It loads `models/pipeline.pkl` and `models/metadata.json` **once** during application startup into application state (`lifespan`).
- **Exact Feature Alignment**: Input features are ordered identically to the Phase 1 training schema (`watch_time_hours`, `sessions_per_week`, `avg_session_mins`, `completion_rate`, `action_pct`, `comedy_pct`, `drama_pct`, `thriller_pct`, `sci_fi_pct`, `weekend_viewing_pct`, `days_since_last_watch`).
- **Deterministic Predictions**: Repeated calls with the same viewer profile always return consistent segment assignments.
- **Euclidean Distance to Centroid**: Calculates $\|\mathbf{x}_{\text{scaled}} - \mathbf{\mu}_c\|_2$ to gauge how closely the viewer profile sits relative to the cluster centroid.
- **Rule-Based Personalization**: Translates the assigned segment into curated, fictional content recommendations with genre-aware refinement.
- **Strict Input Validation**: Enforces non-negative values, ratio bounds $[0, 1]$, and non-empty identifiers, returning clean HTTP 422 JSON errors without exposing internal stack traces.

### Endpoints Overview

| Method | Endpoint | Description | Expected Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Liveness & model readiness probe | `200 OK` (ready) / `503 Unavailable` (model missing) |
| `POST` | `/recommend` | Viewer segmentation & personalized recommendations | `200 OK` / `422 Unprocessable` / `503 Unavailable` |
| `GET` | `/docs` | Interactive Swagger / OpenAPI UI | `200 OK` |
| `GET` | `/redoc` | Interactive ReDoc documentation | `200 OK` |

---

### How to Run the API

1. **Install API Dependencies**:

   ```bash
   pip install -r api/requirements.txt
   ```

2. **Start the API Server**:

   ```bash
   python -m uvicorn api.main:app --reload --port 8000
   ```

   *(or `uvicorn api.main:app --reload`)*

3. **Run the Automated API Test Suite**:

   ```bash
   python api/test_api.py
   ```

---

### Example API Usage

#### 1. Health Check

**Request:**

```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response (HTTP 200):**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### 2. Segment & Recommend (Valid Profile)

**Request:**

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response (HTTP 200):**

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

#### 3. Validation Error (Negative Watch Time or Out-of-Bound Ratio)

**Request:**

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR-8192",
    "watch_time_hours": -10.0,
    "sessions_per_week": 5.0,
    "avg_session_mins": 85.0,
    "completion_rate": 1.5,
    "action_pct": 0.45,
    "comedy_pct": 0.10,
    "drama_pct": 0.15,
    "thriller_pct": 0.20,
    "sci_fi_pct": 0.10,
    "weekend_viewing_pct": 0.65,
    "days_since_last_watch": 2.0
  }'
```

**Response (HTTP 422):**

```json
{
  "error": "Validation Error",
  "message": "Input viewer profile failed validation constraints.",
  "details": [
    {
      "field": "body -> watch_time_hours",
      "message": "Input should be greater than or equal to 0",
      "type": "greater_than_equal"
    },
    {
      "field": "body -> completion_rate",
      "message": "Input should be less than or equal to 1",
      "type": "less_than_equal"
    }
  ]
}
```

---

## 11. Phase 3 — Independent Evaluator Service

Phase 3 introduces an independent validation service (`evaluator/evaluate.py`) that acts as an external test harness to provide objective, audit-grade verification of the ML system and REST API.

### Evaluation Workflow

1. **Liveness Polling**: Waits for `GET /health` to confirm the API is online and the model is fully loaded (`wait_for_api()`).
2. **Representative Profiles (5/5)**: Transmits realistic viewer archetypes across all 5 audience personas, asserting schema adherence and centroid distances.
3. **Negative & Validation Testing (6/6)**: Probes malformed payloads (missing fields, negative durations, out-of-range ratios, type errors) ensuring proper HTTP 422 rejections.
4. **Edge Case Verification (4/4)**: Tests boundary conditions including zero activity (`watch_time_hours=0`, `sessions_per_week=0`), minimum floats, maximum volume, and 100% single-genre dominance.
5. **Prediction Consistency & No-Retraining**: Submits identical requests across multiple iterations to verify exact deterministic stability (confirming read-only inference without retraining).
6. **Empirical Quality Metrics Extraction**: Reads `models/metadata.json` directly to summarize actual silhouette score ($0.3523$), inertia ($3861.80$), and cluster balance metrics.
7. **Audit Report Export**: Writes the complete test results and verified metrics into `metrics.json`.

---

### How to Run the Evaluator

1. **Install Evaluator Dependencies**:

   ```bash
   pip install -r evaluator/requirements.txt
   ```

2. **Ensure the API Server is Running**:

   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

3. **Execute the Evaluator**:

   ```bash
   python evaluator/evaluate.py
   ```

   *Optional: Configure target API address via environment variable:*

   ```bash
   # Windows PowerShell
   $env:API_URL="http://localhost:8000"
   python evaluator/evaluate.py

   # Linux / macOS
   API_URL=http://localhost:8000 python evaluator/evaluate.py
   ```

---

### Verified Evaluation Output (`metrics.json`)

The evaluator produces `metrics.json` at the project root containing actual execution values:

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
