"""
DataPulse – End-to-End Sales Analytics Platform
Tools: Python · Flask · Pandas · NumPy · SQLite/SQL · Matplotlib · Seaborn
       Scikit-Learn · SciPy · Chart.js · REST API · ETL Pipeline · A/B Testing
Deployable on Render
"""

from flask import Flask, jsonify, render_template_string, send_file
import pandas as pd
import numpy as np
from scipy import stats
import sqlite3, io, base64, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  ETL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_raw_data():
    np.random.seed(42)
    n = 600
    dates      = pd.date_range('2023-01-01', '2024-12-31', periods=n)
    categories = ['Electronics', 'Clothing', 'Food & Beverages', 'Books', 'Sports']
    regions    = ['North', 'South', 'East', 'West', 'Central']
    channels   = ['Online', 'Retail', 'Wholesale']

    df = pd.DataFrame({
        'date'         : dates,
        'category'     : np.random.choice(categories, n, p=[0.30, 0.20, 0.25, 0.10, 0.15]),
        'region'       : np.random.choice(regions, n),
        'channel'      : np.random.choice(channels, n, p=[0.50, 0.35, 0.15]),
        'sales'        : np.random.normal(55000, 18000, n),
        'units'        : np.random.randint(15, 600, n),
        'cost'         : np.random.normal(32000, 12000, n),
        'discount_pct' : np.random.uniform(0, 0.30, n),
        'customer_age' : np.random.randint(18, 72, n),
        'rating'       : np.round(np.random.uniform(2.5, 5.0, n), 1),
        'returns'      : np.random.randint(0, 50, n),
    })

    # ── Inject dirty data for cleaning demo ──
    idx = np.random.choice(n, 30, replace=False)
    df.loc[idx[:15], 'sales'] = np.nan
    df.loc[idx[15:], 'cost']  = -999
    return df


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df['sales'] = df['sales'].fillna(df['sales'].median())   # impute nulls
    df = df[df['cost'] > 0].copy()                           # drop invalid cost
    df['sales'] = df['sales'].abs()
    # Feature engineering
    df['gross_revenue']  = df['sales'] * (1 - df['discount_pct'])
    df['profit']         = df['gross_revenue'] - df['cost']
    df['profit_margin']  = (df['profit'] / df['gross_revenue'] * 100).round(2)
    df['revenue_per_unit'] = (df['gross_revenue'] / df['units']).round(2)
    df['month']      = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b')
    df['quarter']    = df['date'].dt.quarter
    df['year']       = df['date'].dt.year
    return df


def load_to_db(df: pd.DataFrame):
    conn = sqlite3.connect('datapulse.db')
    df.to_sql('sales', conn, if_exists='replace', index=True, index_label='id')
    conn.commit()
    conn.close()


def run_etl() -> pd.DataFrame:
    print("▶ Running ETL pipeline …")
    raw   = generate_raw_data()
    clean = clean_data(raw)
    load_to_db(clean)
    print(f"✔ ETL complete — {len(clean)} records loaded.")
    return clean


# ── Boot: run ETL once ──
DF = run_etl()


# ═══════════════════════════════════════════════════════════════════════════════
#  SQL HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def sql(query: str) -> pd.DataFrame:
    conn   = sqlite3.connect('datapulse.db')
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def build_stats():
    sample = DF['profit_margin'].dropna().sample(min(200, len(DF)), random_state=42)
    sw_stat, sw_p = stats.shapiro(sample)
    groups = [g['profit'].values for _, g in DF.groupby('category')]
    f_stat, f_p   = stats.f_oneway(*groups)
    return {
        'normality' : {'stat': round(sw_stat, 4), 'p': round(sw_p, 4), 'normal': bool(sw_p > 0.05)},
        'anova'     : {'f': round(f_stat, 4),     'p': round(f_p, 4),  'sig': bool(f_p < 0.05)},
        'descriptive': DF[['sales','profit','profit_margin','units','rating']].describe().round(2).to_dict(),
    }

STATS = build_stats()


# ═══════════════════════════════════════════════════════════════════════════════
#  MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def build_ml():
    feats  = ['units','discount_pct','cost','customer_age','returns','month','quarter']
    X, y   = DF[feats].fillna(0), DF['gross_revenue']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc     = StandardScaler()
    model  = LinearRegression().fit(sc.fit_transform(Xtr), ytr)
    ypred  = model.predict(sc.transform(Xte))
    coef   = sorted(zip(feats, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    return {
        'r2' : round(r2_score(yte, ypred), 4),
        'mae': round(mean_absolute_error(yte, ypred), 2),
        'features': [{'name': f, 'coef': round(c, 2)} for f, c in coef],
    }

ML = build_ml()


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART GENERATORS (Matplotlib + Seaborn → base64 PNG)
# ═══════════════════════════════════════════════════════════════════════════════

BG, GRID = '#1e293b', '#334155'

def to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor=BG)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


def make_charts():
    plt.rcParams.update({'text.color':'#e2e8f0','axes.labelcolor':'#94a3b8','xtick.color':'#94a3b8','ytick.color':'#94a3b8'})
    charts = {}

    # 1. Revenue Trend – Matplotlib line
    monthly = DF.groupby(['year','month'])['gross_revenue'].sum().reset_index()
    monthly['lbl'] = monthly['year'].astype(str)+'-'+monthly['month'].astype(str).str.zfill(2)
    fig, ax = plt.subplots(figsize=(10,4), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(monthly['lbl'], monthly['gross_revenue']/1e6, color='#38bdf8', lw=2.5, marker='o', ms=4)
    ax.fill_between(monthly['lbl'], monthly['gross_revenue']/1e6, alpha=0.15, color='#38bdf8')
    ax.set_title('Monthly Revenue Trend (₹M)', color='white', fontsize=13)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color(GRID)
    plt.xticks(rotation=45, fontsize=8)
    charts['trend'] = to_b64(fig)

    # 2. Category Profit – Seaborn barplot
    cp = DF.groupby('category')['profit'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8,4), facecolor=BG); ax.set_facecolor(BG)
    colors = ['#f87171','#fb923c','#facc15','#4ade80','#38bdf8']
    ax.barh(cp.index, cp.values/1000, color=colors)
    ax.set_title('Avg Profit by Category (₹K)', color='white', fontsize=13)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color(GRID)
    charts['category'] = to_b64(fig)

    # 3. Correlation Heatmap – Seaborn
    fig, ax = plt.subplots(figsize=(7,5), facecolor=BG); ax.set_facecolor(BG)
    corr = DF[['sales','profit','units','discount_pct','rating']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                linewidths=0.5, annot_kws={'color':'white','size':10})
    ax.set_title('Correlation Matrix', color='white', fontsize=13)
    charts['heatmap'] = to_b64(fig)

    # 4. Profit Margin Distribution – Seaborn histplot + KDE
    fig, ax = plt.subplots(figsize=(8,4), facecolor=BG); ax.set_facecolor(BG)
    sns.histplot(DF['profit_margin'].dropna(), bins=30, color='#818cf8', ax=ax,
                 kde=True, line_kws={'color':'#f472b6','lw':2})
    ax.set_title('Profit Margin Distribution', color='white', fontsize=13)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color(GRID)
    charts['dist'] = to_b64(fig)

    return charts

CHARTS = make_charts()


# ═══════════════════════════════════════════════════════════════════════════════
#  REST API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/kpis')
def api_kpis():
    return jsonify({
        'total_revenue'   : round(DF['gross_revenue'].sum(), 2),
        'total_profit'    : round(DF['profit'].sum(), 2),
        'avg_margin_pct'  : round(DF['profit_margin'].mean(), 2),
        'total_units'     : int(DF['units'].sum()),
        'avg_rating'      : round(DF['rating'].mean(), 2),
        'records'         : len(DF),
    })


@app.route('/api/by-category')
def api_category():
    return jsonify(sql("""
        SELECT category,
               COUNT(*)                        AS transactions,
               ROUND(SUM(gross_revenue),2)     AS revenue,
               ROUND(AVG(profit_margin),2)     AS avg_margin,
               ROUND(SUM(units),0)             AS units
        FROM   sales
        GROUP  BY category
        ORDER  BY revenue DESC
    """).to_dict(orient='records'))


@app.route('/api/by-region')
def api_region():
    return jsonify(sql("""
        WITH totals AS (
            SELECT region,
                   SUM(gross_revenue) AS rev,
                   SUM(profit)        AS pft,
                   COUNT(*)           AS txn
            FROM   sales GROUP BY region
        ),
        overall AS (SELECT SUM(gross_revenue) AS total FROM sales)
        SELECT t.region,
               ROUND(t.rev,2)                       AS revenue,
               ROUND(t.pft,2)                       AS profit,
               t.txn                                AS transactions,
               ROUND(t.rev*100.0/o.total,2)         AS share_pct
        FROM   totals t, overall o
        ORDER  BY revenue DESC
    """).to_dict(orient='records'))


@app.route('/api/monthly')
def api_monthly():
    return jsonify(sql("""
        SELECT year, month,
               ROUND(SUM(gross_revenue),2) AS revenue,
               ROUND(SUM(profit),2)        AS profit,
               COUNT(*)                    AS transactions,
               ROUND(AVG(profit_margin),2) AS avg_margin
        FROM   sales
        GROUP  BY year, month
        ORDER  BY year, month
    """).to_dict(orient='records'))


@app.route('/api/by-channel')
def api_channel():
    return jsonify(sql("""
        SELECT channel,
               ROUND(AVG(gross_revenue),2)        AS avg_revenue,
               ROUND(AVG(discount_pct)*100,2)     AS avg_discount_pct,
               ROUND(AVG(rating),2)               AS avg_rating,
               COUNT(*)                           AS count
        FROM   sales
        GROUP  BY channel
        ORDER  BY avg_revenue DESC
    """).to_dict(orient='records'))


@app.route('/api/statistics')
def api_stats():
    return jsonify(STATS)


@app.route('/api/ml')
def api_ml():
    return jsonify(ML)


@app.route('/api/ab-test')
def api_ab():
    a = DF[DF['channel']=='Online']['profit_margin'].dropna()
    b = DF[DF['channel']=='Retail']['profit_margin'].dropna()
    t, p = stats.ttest_ind(a, b)
    return jsonify({
        'hypothesis'  : 'Online vs Retail profit margin difference',
        'online'      : {'mean': round(a.mean(),2), 'std': round(a.std(),2), 'n': len(a)},
        'retail'      : {'mean': round(b.mean(),2), 'std': round(b.std(),2), 'n': len(b)},
        't_stat'      : round(t,4),
        'p_value'     : round(p,4),
        'significant' : bool(p < 0.05),
        'conclusion'  : 'Significant difference' if p < 0.05 else 'No significant difference',
    })


@app.route('/chart/<name>')
def chart(name):
    if name in CHARTS:
        return send_file(io.BytesIO(base64.b64decode(CHARTS[name])), mimetype='image/png')
    return 'Not found', 404


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD  (single-page, Chart.js + Seaborn charts)
# ═══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DataPulse Analytics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',sans-serif;min-height:100vh}
.hdr{background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:22px 32px;border-bottom:1px solid #1e293b;display:flex;align-items:center;gap:14px}
.hdr h1{font-size:24px;color:#38bdf8;font-weight:700}
.hdr p{color:#94a3b8;font-size:12px;margin-top:3px}
.nav{display:flex;gap:8px;padding:14px 32px;background:#0f172a;border-bottom:1px solid #1e293b;flex-wrap:wrap}
.pill{padding:6px 16px;border-radius:20px;cursor:pointer;font-size:13px;border:1px solid #334155;color:#94a3b8;background:transparent;transition:all .2s}
.pill.on,.pill:hover{background:#38bdf8;color:#0f172a;border-color:#38bdf8;font-weight:700}
.wrap{padding:24px 32px}
.kgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:24px}
.kcard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px;transition:transform .2s,border-color .2s}
.kcard:hover{transform:translateY(-3px);border-color:#38bdf8}
.klbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px}
.kval{font-size:22px;font-weight:700;color:#38bdf8;margin-top:6px}
.ksub{font-size:11px;color:#94a3b8;margin-top:3px}
.cgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px;margin-bottom:20px}
.ccard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px}
.ccard h3{font-size:13px;color:#cbd5e1;margin-bottom:14px;font-weight:600}
.ccard img{width:100%;border-radius:8px}
.tcard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px;margin-bottom:20px;overflow-x:auto}
.tcard h3{font-size:13px;color:#cbd5e1;margin-bottom:12px;font-weight:600}
.qbox{background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:12px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:9px 12px;color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #334155}
td{padding:9px 12px;border-bottom:1px solid #1e293b;color:#cbd5e1}
tr:hover td{background:#334155}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:11px;font-weight:600}
.bg{background:#14532d;color:#4ade80}.bb{background:#1e3a5f;color:#38bdf8}.bo{background:#431407;color:#fb923c}
.sgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;margin-bottom:20px}
.scard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px}
.scard h3{font-size:13px;color:#cbd5e1;margin-bottom:12px;font-weight:600}
.srow{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1e293b;font-size:13px}
.srow:last-child{border-bottom:none}
.sk{color:#94a3b8}.sv{color:#38bdf8;font-weight:600}
.sec{display:none}.sec.on{display:block}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;background:#1e3a5f;color:#38bdf8;font-size:11px;margin:2px}
.acard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px;margin-bottom:14px}
.mth{display:inline-block;padding:2px 8px;border-radius:4px;background:#14532d;color:#4ade80;font-size:11px;font-weight:700;margin-right:8px}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>📊 DataPulse Analytics</h1>
    <p>Python · Flask · Pandas · NumPy · SQLite/SQL · Matplotlib · Seaborn · Scikit-Learn · SciPy · Chart.js · REST API · ETL · A/B Testing</p>
  </div>
</div>
<div class="nav">
  <button class="pill on" onclick="show('ov',this)">📈 Overview</button>
  <button class="pill" onclick="show('ch',this)">📊 Visualizations</button>
  <button class="pill" onclick="show('sq',this)">🗄️ SQL Analytics</button>
  <button class="pill" onclick="show('st',this)">🔬 Statistics & A/B</button>
  <button class="pill" onclick="show('ml',this)">🤖 ML Insights</button>
  <button class="pill" onclick="show('ap',this)">⚡ REST API</button>
</div>
<div class="wrap">

<!-- OVERVIEW -->
<div id="ov" class="sec on">
  <div class="kgrid" id="kgrid"><div class="kcard"><div class="klbl">Loading…</div></div></div>
  <div class="cgrid">
    <div class="ccard"><h3>📈 Revenue Trend — Matplotlib</h3><img src="/chart/trend"></div>
    <div class="ccard"><h3>📊 Avg Profit by Category — Seaborn</h3><img src="/chart/category"></div>
  </div>
  <div class="ccard"><h3>🎯 Channel Performance — Chart.js</h3><canvas id="chnChart" height="80"></canvas></div>
</div>

<!-- CHARTS -->
<div id="ch" class="sec">
  <div class="cgrid">
    <div class="ccard"><h3>🔥 Correlation Matrix — Seaborn Heatmap</h3><img src="/chart/heatmap"></div>
    <div class="ccard"><h3>📉 Profit Margin Distribution — Seaborn + KDE</h3><img src="/chart/dist"></div>
  </div>
  <div class="ccard"><h3>📊 Monthly Revenue vs Profit — Chart.js</h3><canvas id="monChart" height="80"></canvas></div>
</div>

<!-- SQL -->
<div id="sq" class="sec">
  <div class="tcard">
    <h3>🗄️ Sales by Category — GROUP BY + Aggregation</h3>
    <div class="qbox">SELECT category, COUNT(*), SUM(gross_revenue), AVG(profit_margin), SUM(units) FROM sales GROUP BY category ORDER BY revenue DESC</div>
    <table id="tcat"></table>
  </div>
  <div class="tcard">
    <h3>🗺️ Regional Performance — CTE (Common Table Expression)</h3>
    <div class="qbox">WITH totals AS (SELECT region, SUM(gross_revenue) AS rev … GROUP BY region), overall AS (SELECT SUM(…)) SELECT region, revenue, profit, share_pct FROM totals, overall ORDER BY revenue DESC</div>
    <table id="treg"></table>
  </div>
</div>

<!-- STATS -->
<div id="st" class="sec">
  <div class="sgrid">
    <div class="scard"><h3>📐 Normality Test — Shapiro-Wilk</h3><div id="snorm"></div></div>
    <div class="scard"><h3>📊 ANOVA — Profit Difference Across Categories</h3><div id="sanova"></div></div>
    <div class="scard"><h3>🧪 A/B Test — Online vs Retail Margin</h3><div id="sab"></div></div>
  </div>
  <div class="tcard"><h3>📋 Descriptive Statistics</h3><table id="tdesc"></table></div>
</div>

<!-- ML -->
<div id="ml" class="sec">
  <div class="sgrid">
    <div class="scard"><h3>🤖 Model Metrics — Linear Regression</h3><div id="mmet"></div></div>
    <div class="scard" style="grid-column:span 2"><h3>🎯 Feature Importance (Coefficient Magnitude)</h3><canvas id="featChart" height="140"></canvas></div>
  </div>
</div>

<!-- API -->
<div id="ap" class="sec">
  <div class="acard">
    <h3>⚡ REST API Endpoints</h3>
    <div style="margin-top:12px">
      <div class="qbox"><span class="mth">GET</span>/api/kpis — Key performance indicators</div>
      <div class="qbox"><span class="mth">GET</span>/api/by-category — SQL GROUP BY aggregations</div>
      <div class="qbox"><span class="mth">GET</span>/api/by-region — SQL CTE regional analysis</div>
      <div class="qbox"><span class="mth">GET</span>/api/monthly — Time-series revenue data</div>
      <div class="qbox"><span class="mth">GET</span>/api/by-channel — Channel performance</div>
      <div class="qbox"><span class="mth">GET</span>/api/statistics — Statistical test results</div>
      <div class="qbox"><span class="mth">GET</span>/api/ml — ML model metrics & feature importance</div>
      <div class="qbox"><span class="mth">GET</span>/api/ab-test — A/B testing results</div>
    </div>
  </div>
  <div class="acard">
    <h3>🛠️ Full Tech Stack Used</h3>
    <div style="margin-top:10px">
      <span class="tag">Python</span><span class="tag">Flask</span><span class="tag">Pandas</span><span class="tag">NumPy</span>
      <span class="tag">SQLite</span><span class="tag">SQL CTEs</span><span class="tag">Joins & Aggregations</span>
      <span class="tag">Matplotlib</span><span class="tag">Seaborn</span><span class="tag">Chart.js</span>
      <span class="tag">Scikit-Learn</span><span class="tag">SciPy</span><span class="tag">Linear Regression</span>
      <span class="tag">ETL Pipeline</span><span class="tag">Statistical Analysis</span>
      <span class="tag">A/B Testing</span><span class="tag">REST API</span><span class="tag">Data Cleaning</span>
    </div>
  </div>
</div>

</div><!-- /wrap -->

<script>
function show(id,btn){
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('on'));
  document.querySelectorAll('.pill').forEach(p=>p.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  btn.classList.add('on');
}
const fmt=v=>'₹'+(v/1e6).toFixed(2)+'M';

async function loadKPIs(){
  const d=await fetch('/api/kpis').then(r=>r.json());
  document.getElementById('kgrid').innerHTML=`
    <div class="kcard"><div class="klbl">Total Revenue</div><div class="kval">${fmt(d.total_revenue)}</div><div class="ksub">${d.records} transactions</div></div>
    <div class="kcard"><div class="klbl">Total Profit</div><div class="kval">${fmt(d.total_profit)}</div></div>
    <div class="kcard"><div class="klbl">Avg Profit Margin</div><div class="kval">${d.avg_margin_pct}%</div></div>
    <div class="kcard"><div class="klbl">Units Sold</div><div class="kval">${d.total_units.toLocaleString()}</div></div>
    <div class="kcard"><div class="klbl">Avg Rating</div><div class="kval">⭐ ${d.avg_rating}</div></div>`;
}

async function loadChannel(){
  const d=await fetch('/api/by-channel').then(r=>r.json());
  new Chart(document.getElementById('chnChart'),{
    type:'bar',
    data:{labels:d.map(x=>x.channel),datasets:[
      {label:'Avg Revenue',data:d.map(x=>x.avg_revenue),backgroundColor:'#38bdf8aa',borderColor:'#38bdf8',borderWidth:2},
      {label:'Avg Discount %×1000',data:d.map(x=>x.avg_discount_pct*1000),backgroundColor:'#f472b6aa',borderColor:'#f472b6',borderWidth:2}
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#94a3b8'},grid:{color:'#1e293b'}},y:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}}}}
  });
}

async function loadMonthly(){
  const d=await fetch('/api/monthly').then(r=>r.json());
  const lbl=d.map(x=>`${x.year}-${String(x.month).padStart(2,'0')}`);
  new Chart(document.getElementById('monChart'),{
    type:'line',
    data:{labels:lbl,datasets:[
      {label:'Revenue',data:d.map(x=>x.revenue),borderColor:'#38bdf8',backgroundColor:'#38bdf815',fill:true,tension:0.4},
      {label:'Profit', data:d.map(x=>x.profit), borderColor:'#4ade80',backgroundColor:'#4ade8015',fill:true,tension:0.4}
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:12},grid:{color:'#1e293b'}},y:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}}}}
  });
}

async function loadSQL(){
  const cats=await fetch('/api/by-category').then(r=>r.json());
  document.getElementById('tcat').innerHTML=`
    <tr><th>Category</th><th>Transactions</th><th>Revenue</th><th>Avg Margin</th><th>Units</th></tr>
    ${cats.map(d=>`<tr><td>${d.category}</td><td>${d.transactions}</td><td>${fmt(d.revenue)}</td><td><span class="badge bg">${d.avg_margin}%</span></td><td>${Number(d.units).toLocaleString()}</td></tr>`).join('')}`;
  const reg=await fetch('/api/by-region').then(r=>r.json());
  document.getElementById('treg').innerHTML=`
    <tr><th>Region</th><th>Revenue</th><th>Profit</th><th>Transactions</th><th>Share %</th></tr>
    ${reg.map(d=>`<tr><td>${d.region}</td><td>${fmt(d.revenue)}</td><td>${fmt(d.profit)}</td><td>${d.transactions}</td><td><span class="badge bb">${d.share_pct}%</span></td></tr>`).join('')}`;
}

async function loadStats(){
  const s=await fetch('/api/statistics').then(r=>r.json());
  const ab=await fetch('/api/ab-test').then(r=>r.json());
  document.getElementById('snorm').innerHTML=`
    <div class="srow"><span class="sk">W Statistic</span><span class="sv">${s.normality.stat}</span></div>
    <div class="srow"><span class="sk">P-Value</span><span class="sv">${s.normality.p}</span></div>
    <div class="srow"><span class="sk">Normal?</span><span class="sv">${s.normality.normal?'✅ Yes':'❌ No'}</span></div>`;
  document.getElementById('sanova').innerHTML=`
    <div class="srow"><span class="sk">F-Statistic</span><span class="sv">${s.anova.f}</span></div>
    <div class="srow"><span class="sk">P-Value</span><span class="sv">${s.anova.p}</span></div>
    <div class="srow"><span class="sk">Significant?</span><span class="sv">${s.anova.sig?'✅ Yes':'❌ No'}</span></div>`;
  document.getElementById('sab').innerHTML=`
    <div class="srow"><span class="sk">Online Margin</span><span class="sv">${ab.online.mean}%</span></div>
    <div class="srow"><span class="sk">Retail Margin</span><span class="sv">${ab.retail.mean}%</span></div>
    <div class="srow"><span class="sk">T-Statistic</span><span class="sv">${ab.t_stat}</span></div>
    <div class="srow"><span class="sk">P-Value</span><span class="sv">${ab.p_value}</span></div>
    <div class="srow"><span class="sk">Result</span><span class="sv" style="font-size:11px">${ab.conclusion}</span></div>`;
  const desc=s.descriptive;
  const cols=Object.keys(desc);
  const rows=Object.keys(desc[cols[0]]);
  document.getElementById('tdesc').innerHTML=`
    <tr><th>Metric</th>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>
    ${rows.map(r=>`<tr><td>${r}</td>${cols.map(c=>`<td>${Object.values(desc[c])[rows.indexOf(r)]??'-'}</td>`).join('')}</tr>`).join('')}`;
}

async function loadML(){
  const d=await fetch('/api/ml').then(r=>r.json());
  document.getElementById('mmet').innerHTML=`
    <div class="srow"><span class="sk">Algorithm</span><span class="sv">Linear Regression</span></div>
    <div class="srow"><span class="sk">R² Score</span><span class="sv">${d.r2}</span></div>
    <div class="srow"><span class="sk">Mean Abs Error</span><span class="sv">₹${d.mae.toLocaleString()}</span></div>
    <div class="srow"><span class="sk">Target Variable</span><span class="sv">Gross Revenue</span></div>
    <div class="srow"><span class="sk">Feature Count</span><span class="sv">${d.features.length}</span></div>`;
  new Chart(document.getElementById('featChart'),{
    type:'bar',
    data:{labels:d.features.map(f=>f.name),datasets:[{
      label:'Coefficient',data:d.features.map(f=>f.coef),
      backgroundColor:d.features.map(f=>f.coef>0?'#4ade80aa':'#f87171aa'),
      borderColor:d.features.map(f=>f.coef>0?'#4ade80':'#f87171'),borderWidth:2}]},
    options:{responsive:true,indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}},y:{ticks:{color:'#94a3b8'},grid:{color:'#1e293b'}}}}
  });
}

loadKPIs();loadChannel();loadMonthly();loadSQL();loadStats();loadML();
</script>
</body>
</html>"""

@app.route('/')
def index():
    return HTML

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
