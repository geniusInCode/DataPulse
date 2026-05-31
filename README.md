# 📊 DataPulse – End-to-End Sales Analytics Platform

A full-stack data analytics platform demonstrating ETL pipelines, SQL analytics, statistical testing, machine learning, and interactive dashboards.

## 🛠️ Tech Stack
- **Python** – Core language
- **Flask** – Web framework & REST API
- **Pandas & NumPy** – Data processing & ETL
- **SQLite + SQL** – Joins, Aggregations, CTEs
- **Matplotlib & Seaborn** – Static visualizations
- **Scikit-Learn** – Linear Regression ML model
- **SciPy** – Statistical analysis (Shapiro-Wilk, ANOVA, t-test)
- **Chart.js** – Interactive frontend charts

## 🚀 Features
- **ETL Pipeline** – Raw data generation → cleaning → SQLite load
- **SQL Analytics** – GROUP BY, CTEs, window-style queries
- **Statistical Analysis** – Normality test, ANOVA, A/B testing
- **ML Predictions** – Revenue prediction with feature importance
- **REST API** – 8 JSON endpoints
- **Interactive Dashboard** – 6-tab analytics UI

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

## ☁️ Deploy on Render
1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Click Deploy

## 📡 API Endpoints
| Endpoint | Description |
|---|---|
| `GET /api/kpis` | Key performance indicators |
| `GET /api/by-category` | SQL GROUP BY aggregation |
| `GET /api/by-region` | SQL CTE regional analysis |
| `GET /api/monthly` | Time-series trend data |
| `GET /api/by-channel` | Channel performance |
| `GET /api/statistics` | Statistical test results |
| `GET /api/ml` | ML metrics & feature importance |
| `GET /api/ab-test` | A/B testing: Online vs Retail |
