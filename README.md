# BI System — Machine Learning Analytics for ERP

A production-grade **Business Intelligence microservice** that connects to an ERP database and exposes ML-powered analytics — product scoring, employee performance evaluation, sales forecasting, recommendation engine, and market basket analysis — through a clean REST API.

Built with **Flask + scikit-learn + Facebook Prophet** and deployed via **Docker + Nginx**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask 3.0"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Facebook%20Prophet-0B3D91?style=for-the-badge&logo=facebook&logoColor=white" alt="Facebook Prophet"/>
  <img src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white" alt="SQLAlchemy"/>
  <img src="https://img.shields.io/badge/MariaDB-003545?style=for-the-badge&logo=mariadb&logoColor=white" alt="MariaDB"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white" alt="Nginx"/>
</p>

---

## ✨ Features

| Module | Technique | Endpoint |
|---|---|---|
| 📦 **Product Performance Analysis** | K-Means clustering + weighted scoring | `POST /api/product-analysis` |
| 👥 **Employee Performance Evaluation** | K-Means + Isolation Forest anomaly detection | `POST /api/analyze` |
| 📈 **Sales Forecast** | Linear Regression with train/test split | `GET /api/sales-forecast` |
| 🔮 **Prophet Forecast** | Facebook Prophet (seasonality + trend) | `GET /api/prophet-forecast` |
| 🛒 **Product Recommendations** | KNN content-based + collaborative filtering + trending | `POST /api/recommendations/products` |
| 🧺 **Market Basket Analysis** | Apriori algorithm (frequent itemsets + association rules) | `GET /api/product-combos` |

---

## 🖼️ Dashboard Previews

Sample dashboards rendered from each endpoint (synthetic data; generators in `docs/tools/*.py`, full write-ups in `docs/models/`):

**Module 1 — Product Performance Analysis**
<p align="center"><img src="docs/images/01-product-analysis-dashboard.png" alt="Product Analysis dashboard" width="100%"/></p>

**Module 2 — Employee Performance Evaluation**
<p align="center"><img src="docs/images/02-employee-performance-dashboard.png" alt="Employee Performance dashboard" width="100%"/></p>

**Module 3 — Sales Forecast (Linear Regression)**
<p align="center"><img src="docs/images/03-sales-forecast-dashboard.png" alt="Sales Forecast dashboard" width="100%"/></p>

**Module 4 — Sales Forecast (Facebook Prophet)**
<p align="center"><img src="docs/images/04-prophet-forecast-dashboard.png" alt="Prophet Forecast dashboard" width="100%"/></p>

**Module 5 — Product Recommendations**
<p align="center"><img src="docs/images/05-product-recommendations-dashboard.png" alt="Product Recommendations dashboard" width="100%"/></p>

**Module 6 — Market Basket Analysis (Apriori)**
<p align="center"><img src="docs/images/06-product-combos-dashboard.png" alt="Market Basket Analysis dashboard" width="100%"/></p>

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│   Client    │────▶│  Nginx       │────▶│  Flask App    │
│ (Dashboard) │     │  (Reverse    │     │  (BI Service) │
└─────────────┘     │   Proxy)     │     └───────┬───────┘
                    └──────────────┘             │
                                          ┌──────▼───────┐
                                          │  MariaDB     │
                                          │  (ERP DB)    │
                                          └──────────────┘
```

- 🌐 **Nginx** — reverse proxy, request routing, timeout tuning
- 🧠 **Flask App** — 6 route blueprints, lazy model loading, joblib persistence
- 🗄️ **MariaDB** — shared external ERP database (not managed by this container)
- 💾 **Trained models** saved as `.joblib` files under `ml_models/`

---

## 🧰 Tech Stack

- **Backend:** 🐍 Python 3.11, ⚗️ Flask 3.0, Flask-SQLAlchemy, Flask-CORS
- **ML/DL:** 📐 scikit-learn, pandas, numpy, mlxtend (Apriori), Facebook Prophet
- **Infrastructure:** 🐳 Docker, Docker Compose, Nginx
- **Storage:** 🗄️ PyMySQL, joblib (model serialization)

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Access to a MariaDB/MySQL ERP database

### 1. Clone & Configure

```bash
git clone <repo-url>
cd bi-system
```

Edit `flask-app/.env` to point to your database:

```env
DB_HOST=your_mariadb_host
DB_PORT=3306
DB_NAME=laravel_db
DB_USER=root
DB_PASSWORD=root
```

Ensure the network `erp-system_default` exists (or change `docker-compose.yml`):

```bash
docker network create erp-system_default
```

### 2. Run

```bash
docker compose up -d
```

| Service | Port | URL |
|---|---|---|
| Flask API | `5001` | `http://localhost:5001` |
| Nginx | `5002` | `http://localhost:5002` |

### 3. Check Health

```bash
curl http://localhost:5001/
# {"status":"ok","service":"BI System","endpoints":[...]}
```

---

## 📡 API Reference

### `POST /api/product-analysis`
Analyzes all active products using K-Means clustering + weighted scoring (revenue, stock ratio, sales velocity, profit margin).

```json
// Response
{
  "status": "success",
  "data": [{"product_id": 1, "name": "...", "performance_score": 85.3, "performance_tier": "High", ...}],
  "summary": {"total_products": 50, "avg_score": 62.4, "high_tier": 12, ...},
  "model_version": "2.0"
}
```

### `POST /api/analyze`
Evaluates employee performance based on task completion rates, task velocity, and identifies anomalies (Isolation Forest).

### `GET /api/sales-forecast`
Linear regression forecast for next month's revenue. Returns historical monthly aggregated sales + prediction with confidence interval.

### `GET /api/prophet-forecast`
Facebook Prophet forecast. Parameters: `?periods=12` (1–60 months). Returns historical fit, future forecast, trend decomposition, and summary metrics.

### `POST /api/recommendations/products`
Multiple recommendation strategies via JSON body:

```json
{
  "strategy": "popular|trending|content_based|collaborative|personalized",
  "product_id": 123,   // for content_based
  "user_id": 456,       // for collaborative / personalized
  "limit": 10
}
```

### `POST /api/recommendations/train`
Manually retrain the recommendation models (KNN + collaborative filtering).

### `GET /api/product-combos`
Runs Apriori algorithm on completed sales transactions. Returns frequent itemsets and association rules (lift, confidence, support).

---

## 📂 Project Structure

```
bi-system/
├── docker-compose.yml          # Service orchestration
├── nginx/
│   └── default.conf            # Reverse proxy config
├── flask-app/
│   ├── Dockerfile              # Python 3.11 + Prophet + dependencies
│   ├── requirements.txt
│   ├── .env                    # DB credentials (not committed)
│   ├── app.py                  # Flask app factory, blueprint registration
│   ├── models/
│   │   └── __init__.py         # All SQLAlchemy ORM models (15 tables)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── employee_performance.py    # K-Means + Isolation Forest
│   │   ├── product_analysis.py        # K-Means + weighted scoring
│   │   ├── product_recommendations.py # KNN + collaborative + trending
│   │   ├── product_combos.py          # Apriori association rules
│   │   ├── sales_forecast.py          # Linear Regression
│   │   └── prophet_forecast.py        # Facebook Prophet
│   └── ml_models/              # Trained .joblib models (gitignored)
├── docs/                       # Module documentation, dashboard images, HTML export
│   ├── models/                 # In-depth per-module write-ups (Markdown)
│   ├── images/                 # Generated demo dashboard PNGs
│   ├── html/                   # HTML export of the Markdown docs
│   └── tools/                  # Demo dashboard generators + md_to_html.py
├── README.md
├── AGENTS.md                   # AI agent onboarding guide
└── case-study.md               # Comprehensive case study
```

---

## 🏷️ Model Versioning

All endpoints return `"model_version": "2.0"`. Models retrain on every request (except recommendation models which use lazy loading). Trained models are persisted to `ml_models/*.joblib`.

---

## 📜 License

MIT
