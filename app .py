import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import hashlib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Talent Engine | محرك المواهب",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Cairo:wght@300;400;600;700&display=swap');

:root {
    --bg-base:        #080c14;
    --bg-card:        #0e1520;
    --bg-card2:       #111a28;
    --bg-sidebar:     #060b12;
    --border:         #1e2d42;
    --border-glow:    #1e4a8a;
    --accent-blue:    #2b7fff;
    --accent-cyan:    #00d4ff;
    --accent-purple:  #8b5cf6;
    --accent-green:   #10b981;
    --accent-orange:  #f59e0b;
    --accent-red:     #ef4444;
    --text-primary:   #e8f0ff;
    --text-secondary: #7a93b5;
    --text-muted:     #3d5475;
    --gradient-hero:  linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #091427 100%);
    --gradient-card:  linear-gradient(145deg, #0e1520 0%, #111d30 100%);
    --gradient-accent:linear-gradient(135deg, #2b7fff 0%, #8b5cf6 100%);
    --shadow-card:    0 4px 24px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.03) inset;
    --shadow-glow:    0 0 40px rgba(43,127,255,0.15);
    --sw:             270px;
    --radius:         14px;
    --radius-sm:      8px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Cairo', 'Syne', sans-serif !important;
}

/* إخفاء Streamlit sidebar الأصلي */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] {
    display: none !important;
    width: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
}

@media (min-width: 993px) {
    .block-container { margin-left: var(--sw) !important; }
}
@media (max-width: 992px) {
    .block-container { margin-left: 0 !important; padding: 1rem !important; padding-top: 68px !important; }
    .page-hero h1  { font-size: 1.3rem !important; }
    .page-hero p   { max-width: 100% !important; font-size: 0.82rem !important; }
    .stat-card .stat-value { font-size: 1.5rem !important; }
}

/* ══ CUSTOM SIDEBAR ══ */
#cSidebar {
    position: fixed;
    top: 0; left: 0;
    width: var(--sw);
    height: 100vh;
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
    z-index: 9999;
    overflow-y: auto; overflow-x: hidden;
    display: flex; flex-direction: column;
    padding: 1.4rem 1.2rem 1.2rem;
    transition: transform 0.35s cubic-bezier(0.4,0,0.2,1);
    will-change: transform;
}
#cSidebar::-webkit-scrollbar { width: 3px; }
#cSidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

@media (min-width: 993px) {
    #cSidebar { transform: translateX(0) !important; box-shadow: none; }
    #hBtn, #sbOverlay { display: none !important; }
}
@media (max-width: 992px) {
    #cSidebar { transform: translateX(-100%); box-shadow: 8px 0 32px rgba(0,0,0,0.7); }
    #cSidebar.open { transform: translateX(0); }
}

/* ══ HAMBURGER ══ */
#hBtn {
    display: none;
    position: fixed;
    top: 13px; left: 13px;
    z-index: 10000;
    width: 42px; height: 42px;
    background: var(--gradient-accent);
    border: none; border-radius: 10px;
    cursor: pointer;
    flex-direction: column; align-items: center; justify-content: center;
    gap: 5px; padding: 0;
    box-shadow: 0 2px 14px rgba(43,127,255,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
#hBtn:hover { transform: scale(1.06); box-shadow: 0 4px 22px rgba(43,127,255,0.6); }
#hBtn .hb {
    display: block; width: 19px; height: 2px;
    background: #fff; border-radius: 2px;
    transition: transform 0.3s cubic-bezier(0.4,0,0.2,1), opacity 0.25s;
}
#hBtn.open .hb:nth-child(1) { transform: translateY(7px) rotate(45deg); }
#hBtn.open .hb:nth-child(2) { opacity: 0; transform: scaleX(0); }
#hBtn.open .hb:nth-child(3) { transform: translateY(-7px) rotate(-45deg); }
@media (max-width: 992px) { #hBtn { display: flex !important; } }

/* ══ OVERLAY ══ */
#sbOverlay {
    display: none; position: fixed; inset: 0;
    background: rgba(4,8,18,0.78); z-index: 9998;
    backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px);
    opacity: 0; transition: opacity 0.3s ease;
}
#sbOverlay.vis { display: block; }
#sbOverlay.act { opacity: 1; }

/* ══ Sidebar inner components ══ */
.sb-logo { font-family:'Syne',sans-serif; font-size:1rem; font-weight:800; background:var(--gradient-accent); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.sb-ver  { font-size:0.65rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace; margin-top:2px; }
.sb-div  { height:1px; background:var(--border); margin:1rem 0; opacity:.6; }
.sb-ucard { background:rgba(43,127,255,0.05); border:1px solid var(--border); border-radius:10px; padding:.8rem 1rem; margin-bottom:1.1rem; }
.sb-uname { font-family:'Syne',sans-serif; font-weight:700; font-size:.86rem; color:var(--text-primary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sb-uemail{ font-size:.65rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sb-nlbl  { font-size:.62rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.45rem; }
.sb-nbtn  { display:flex; align-items:center; gap:.6rem; width:100%; padding:.6rem .85rem; background:transparent; border:1px solid transparent; border-radius:9px; color:var(--text-secondary); font-family:'Syne',sans-serif; font-size:.83rem; font-weight:600; cursor:pointer; margin-bottom:.22rem; transition:all .18s; text-align:left; }
.sb-nbtn:hover  { background:rgba(43,127,255,.08); border-color:rgba(43,127,255,.2); color:var(--text-primary); }
.sb-nbtn.active { background:rgba(43,127,255,.12); border-color:rgba(43,127,255,.35); color:var(--accent-cyan); }
.sb-nbtn .ico   { font-size:.95rem; width:18px; text-align:center; flex-shrink:0; }
.sb-lbtn  { display:flex; align-items:center; gap:.6rem; width:100%; padding:.6rem .85rem; background:rgba(239,68,68,.06); border:1px solid rgba(239,68,68,.2); border-radius:9px; color:#fca5a5; font-family:'Syne',sans-serif; font-size:.83rem; font-weight:600; cursor:pointer; transition:all .18s; }
.sb-lbtn:hover  { background:rgba(239,68,68,.14); border-color:rgba(239,68,68,.4); }
.sb-foot  { margin-top:auto; padding-top:1rem; text-align:center; font-size:.6rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace; line-height:1.6; }

/* ══ Cards ══ */
.engine-card { background:var(--gradient-card); border:1px solid var(--border); border-radius:var(--radius); padding:1.5rem; margin-bottom:1rem; box-shadow:var(--shadow-card); transition:border-color .2s,box-shadow .2s; position:relative; overflow:hidden; }
.engine-card::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; background:var(--gradient-accent); opacity:.7; }
.engine-card:hover { border-color:var(--border-glow); box-shadow:var(--shadow-glow); }
.stat-card { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:1.3rem 1.1rem; text-align:center; position:relative; overflow:hidden; transition:all .3s; }
.stat-card:hover { transform:translateY(-3px); border-color:var(--border-glow); box-shadow:var(--shadow-glow); }
.stat-card .stat-icon  { font-size:1.7rem; margin-bottom:.35rem; display:block; }
.stat-card .stat-value { font-family:'JetBrains Mono',monospace; font-size:1.9rem; font-weight:700; background:var(--gradient-accent); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; line-height:1.1; }
.stat-card .stat-label { font-family:'Syne',sans-serif; font-size:.7rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:.08em; margin-top:.3rem; }
.stat-card .stat-delta { font-size:.7rem; margin-top:.25rem; font-family:'JetBrains Mono',monospace; }
.delta-pos { color:var(--accent-green); } .delta-neg { color:var(--accent-red); }
.page-hero { background:var(--gradient-hero); border:1px solid var(--border); border-radius:var(--radius); padding:1.8rem 2rem; margin-bottom:1.8rem; position:relative; overflow:hidden; }
.page-hero::after { content:''; position:absolute; right:-60px;top:-60px; width:220px;height:220px; background:radial-gradient(circle,rgba(43,127,255,.12) 0%,transparent 70%); border-radius:50%; }
.page-hero h1 { font-family:'Syne',sans-serif; font-size:1.75rem; font-weight:800; background:linear-gradient(135deg,#e8f0ff 30%,var(--accent-cyan) 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:.35rem; }
.page-hero p  { color:var(--text-secondary); font-size:.9rem; max-width:65%; }
.badge { display:inline-block; padding:.17rem .6rem; border-radius:100px; font-size:.68rem; font-family:'JetBrains Mono',monospace; font-weight:600; letter-spacing:.04em; }
.badge-blue   { background:rgba(43,127,255,.15);  color:var(--accent-blue);   border:1px solid rgba(43,127,255,.3); }
.badge-green  { background:rgba(16,185,129,.15); color:var(--accent-green);  border:1px solid rgba(16,185,129,.3); }
.badge-orange { background:rgba(245,158,11,.15); color:var(--accent-orange); border:1px solid rgba(245,158,11,.3); }
.badge-purple { background:rgba(139,92,246,.15); color:var(--accent-purple); border:1px solid rgba(139,92,246,.3); }
.skill-tag     { display:inline-block; background:rgba(43,127,255,.1); border:1px solid rgba(43,127,255,.25); color:var(--accent-cyan); border-radius:6px; padding:.17rem .52rem; font-size:.68rem; font-family:'JetBrains Mono',monospace; margin:2px; }
.skill-gap-tag { display:inline-block; background:rgba(245,158,11,.1); border:1px solid rgba(245,158,11,.25); color:var(--accent-orange); border-radius:6px; padding:.17rem .52rem; font-size:.68rem; font-family:'JetBrains Mono',monospace; margin:2px; }
.section-title { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:var(--text-primary); margin-bottom:1rem; display:flex; align-items:center; gap:.5rem; }
.section-title::after { content:''; flex:1; height:1px; background:linear-gradient(to right,var(--border),transparent); }
.risk-high   { color:var(--accent-red);    font-weight:700; }
.risk-medium { color:var(--accent-orange); font-weight:700; }
.risk-low    { color:var(--accent-green);  font-weight:700; }
.progress-bar-wrap { background:var(--border); border-radius:100px; height:7px; overflow:hidden; margin-top:.3rem; }
.progress-bar-fill { height:100%; border-radius:100px; background:var(--gradient-accent); transition:width .6s ease; }
.job-card { background:var(--bg-card2); border:1px solid var(--border); border-radius:var(--radius); padding:1.1rem 1.2rem; margin-bottom:.75rem; transition:all .2s; }
.job-card:hover { border-color:var(--accent-blue); }
.job-card h4 { font-family:'Syne',sans-serif; font-size:.93rem; font-weight:700; color:var(--accent-cyan); margin-bottom:.25rem; }
.alert-box { padding:.72rem 1rem; border-radius:var(--radius-sm); font-family:'Cairo',sans-serif; font-size:.84rem; margin-bottom:1rem; }
.alert-error   { background:rgba(239,68,68,.1);  border:1px solid rgba(239,68,68,.3);  color:#fca5a5; }
.alert-success { background:rgba(16,185,129,.1); border:1px solid rgba(16,185,129,.3); color:#6ee7b7; }
.alert-info    { background:rgba(43,127,255,.1); border:1px solid rgba(43,127,255,.3); color:#93c5fd; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg-base); }
::-webkit-scrollbar-thumb { background:var(--border-glow); border-radius:3px; }
#MainMenu, footer, header { visibility:hidden !important; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DB_PATH = "talent_engine.db"

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db(); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, full_name TEXT, role TEXT DEFAULT 'user',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP, last_login TEXT)""")
    c.execute("INSERT OR IGNORE INTO users (email,password_hash,full_name,role) VALUES (?,?,?,?)",
              ("kareemeltemsah7@gmail.com", hashlib.sha256(b"temsah1").hexdigest(), "Kareem El-Temsah", "admin"))
    conn.commit(); conn.close()

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def login_user(email, password):
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT id,email,full_name,role FROM users WHERE email=? AND password_hash=?", (email, hash_pw(password)))
    row = c.fetchone()
    if row:
        c.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.now().isoformat(), row[0]))
        conn.commit()
    conn.close(); return row

def register_user(email, password, full_name):
    conn = get_db(); c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email,password_hash,full_name,role) VALUES (?,?,?,?)",
                  (email, hash_pw(password), full_name, "user"))
        conn.commit(); conn.close(); return True, "تم إنشاء الحساب بنجاح!"
    except sqlite3.IntegrityError:
        conn.close(); return False, "البريد الإلكتروني مسجل مسبقاً."
    except Exception as e:
        conn.close(); return False, str(e)

def get_all_users():
    conn = get_db()
    df = pd.read_sql("SELECT id,email,full_name,role,created_at,last_login FROM users", conn)
    conn.close(); return df

def delete_user(uid, cur_uid):
    if uid == cur_uid: return False, "لا يمكنك حذف حسابك الحالي."
    conn = get_db(); c = conn.cursor()
    c.execute("DELETE FROM users WHERE id=?", (uid,))
    ok = c.rowcount > 0; conn.commit(); conn.close()
    return ok, "تم حذف المستخدم." if ok else "المستخدم غير موجود."

def promote_to_admin(uid):
    conn = get_db(); c = conn.cursor()
    c.execute("UPDATE users SET role='admin' WHERE id=?", (uid,))
    ok = c.rowcount > 0; conn.commit(); conn.close()
    return ok, "تمت الترقية إلى Admin."

def demote_to_user(uid, cur_uid, cur_role):
    if uid == cur_uid and cur_role == "admin": return False, "لا يمكنك خفض دورك بنفسك."
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
    if c.fetchone()[0] <= 1: conn.close(); return False, "يجب أن يبقى مدير واحد على الأقل."
    c.execute("UPDATE users SET role='user' WHERE id=?", (uid,))
    ok = c.rowcount > 0; conn.commit(); conn.close()
    return ok, "تم خفض الدور إلى User."

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
DEPARTMENTS = ["Engineering","Product","Data Science","Design","Sales","Marketing",
               "Finance","HR","Operations","Customer Success","Legal","Security","DevOps","QA","Research"]
SALARY_TIERS = ["Tier 1 (Low)","Tier 2 (Mid)","Tier 3 (Senior)","Tier 4 (Lead)","Tier 5 (Executive)"]
SKILLS_POOL = {
    "Engineering":      ["Python","Java","Go","Kubernetes","AWS","React","TypeScript","PostgreSQL","Redis","gRPC","Docker","Terraform","CI/CD","GraphQL"],
    "Product":          ["Roadmapping","PRD Writing","A/B Testing","User Research","JIRA","Analytics","OKRs","Stakeholder Management","SQL","Figma"],
    "Data Science":     ["Python","R","Machine Learning","Deep Learning","SQL","Spark","Tableau","Statistics","NLP","Computer Vision","MLflow","Pandas"],
    "Design":           ["Figma","UI Design","UX Research","Prototyping","Motion Design","Brand Identity","Accessibility","Design Systems","Sketch"],
    "Sales":            ["CRM","Salesforce","Lead Generation","Negotiation","Account Management","Forecasting","Cold Outreach","SaaS Sales","Demos"],
    "Marketing":        ["SEO","Content Strategy","Paid Media","Email Marketing","Analytics","HubSpot","Copywriting","Brand Management","Social Media"],
    "Finance":          ["Financial Modeling","Excel","Python","Risk Analysis","Budgeting","FP&A","GAAP","Bloomberg","QuickBooks","SQL"],
    "HR":               ["Recruiting","HRIS","Compensation Design","L&D","Performance Management","Employment Law","ATS","Onboarding","Culture"],
    "Operations":       ["Process Optimization","Six Sigma","Supply Chain","ERP","Vendor Management","Data Analysis","Lean","Project Management"],
    "Customer Success": ["Onboarding","Churn Prevention","CRM","Zendesk","SLAs","NPS","Upselling","Product Adoption","Communication"],
    "Legal":            ["Contract Review","IP Law","Privacy (GDPR)","Employment Law","M&A","Compliance","Regulatory Affairs"],
    "Security":         ["Penetration Testing","SIEM","Zero Trust","SOC","Cloud Security","Python","Incident Response","Compliance","Threat Modeling"],
    "DevOps":           ["Kubernetes","Terraform","AWS","Ansible","CI/CD","Docker","Monitoring","Linux","Python","Infrastructure as Code"],
    "QA":               ["Test Automation","Selenium","Pytest","Manual Testing","Performance Testing","API Testing","JIRA","Bug Reporting"],
    "Research":         ["Research Design","Statistics","Python","R","Literature Review","Grant Writing","LaTeX","NLP","Academic Publishing"],
}
INTERNAL_ROLES = [
    {"title":"Senior ML Engineer",         "dept":"Data Science",    "skills":["Python","Machine Learning","Deep Learning","MLflow","AWS","SQL","Statistics"]},
    {"title":"Product Manager II",          "dept":"Product",         "skills":["Roadmapping","A/B Testing","SQL","OKRs","Analytics","Stakeholder Management"]},
    {"title":"DevOps Lead",                 "dept":"DevOps",          "skills":["Kubernetes","Terraform","AWS","Docker","CI/CD","Python","Monitoring"]},
    {"title":"UX Research Lead",            "dept":"Design",          "skills":["UX Research","Figma","User Research","Prototyping","Design Systems"]},
    {"title":"Data Engineer",               "dept":"Data Science",    "skills":["Python","Spark","SQL","Pandas","AWS","Airflow","PostgreSQL"]},
    {"title":"Head of Customer Success",    "dept":"Customer Success","skills":["Churn Prevention","NPS","CRM","Upselling","SLAs","Onboarding"]},
    {"title":"Security Architect",          "dept":"Security",        "skills":["Zero Trust","Cloud Security","Threat Modeling","SIEM","Python","Compliance"]},
    {"title":"Frontend Engineer III",       "dept":"Engineering",     "skills":["React","TypeScript","GraphQL","CSS","Testing","Figma","Performance"]},
    {"title":"Growth Marketing Manager",    "dept":"Marketing",       "skills":["Paid Media","Analytics","A/B Testing","SEO","Content Strategy","HubSpot"]},
    {"title":"Finance Business Partner",    "dept":"Finance",         "skills":["Financial Modeling","FP&A","Python","SQL","Excel","Budgeting","Analytics"]},
    {"title":"Platform Engineer",           "dept":"DevOps",          "skills":["Kubernetes","Terraform","Go","AWS","CI/CD","gRPC","Infrastructure as Code"]},
    {"title":"Research Scientist",          "dept":"Research",        "skills":["Python","Deep Learning","NLP","Statistics","PyTorch","Research Design","LaTeX"]},
    {"title":"Enterprise Account Executive","dept":"Sales",           "skills":["SaaS Sales","CRM","Salesforce","Negotiation","Forecasting","Account Management"]},
    {"title":"HR Business Partner",         "dept":"HR",              "skills":["Performance Management","L&D","Compensation Design","Recruiting","Culture","HRIS"]},
    {"title":"Backend Engineer Staff",      "dept":"Engineering",     "skills":["Go","Python","PostgreSQL","Redis","gRPC","AWS","Kubernetes","Architecture"]},
]

@st.cache_data(show_spinner=False)
def generate_employee_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    depts = rng.choice(DEPARTMENTS, size=n)
    tenure = rng.integers(1,145,size=n).astype(float)
    tn = (tenure - tenure.min()) / (tenure.max() - tenure.min())
    sp = rng.dirichlet(np.ones(5), size=n)
    for i in range(n):
        sp[i,0] = max(0.05, sp[i,0]*(1-tn[i])); sp[i,4] = sp[i,4]*(0.5+tn[i])
    sp /= sp.sum(axis=1, keepdims=True)
    sidx = np.array([rng.choice(5,p=r) for r in sp])
    avg_hours = np.clip(140+rng.normal(0,20,n)+(4-sidx)*rng.uniform(0,8,n), 120, 310).round(1)
    satisfaction = np.clip(0.5-0.0015*(avg_hours-160)+0.05*sidx+0.002*tn*10+rng.normal(0,0.12,n), 0.1, 1.0).round(3)
    evaluation = np.clip(rng.normal(0.68,0.12,n)+0.04*sidx/4, 0.2, 1.0).round(3)
    skills_list = []
    for d in depts:
        pool = SKILLS_POOL.get(d, SKILLS_POOL["Engineering"])
        k = rng.integers(3,min(len(pool)+1,8))
        skills_list.append(", ".join(rng.choice(pool,size=k,replace=False).tolist()))
    cs = (0.30*(1-satisfaction)+0.25*(avg_hours-140)/170+0.15*(1-evaluation)
          +0.20*(1-sidx/4)+0.10*(1-tn)+rng.normal(0,0.05,n))
    return pd.DataFrame({
        "Employee_ID":           [f"EMP-{10000+i}" for i in range(n)],
        "Department":            depts, "Tenure_Months": tenure.astype(int),
        "Satisfaction_Score":    satisfaction, "Avg_Monthly_Hours": avg_hours,
        "Last_Evaluation_Score": evaluation, "Salary_Tier": [SALARY_TIERS[i] for i in sidx],
        "Salary_Tier_Num":       sidx+1, "Skills": skills_list,
        "Churn_Risk":            (cs > np.percentile(cs,70)).astype(int),
        "Churn_Score_Raw":       np.clip(cs,0,1).round(4),
    })

@st.cache_resource(show_spinner=False)
def train_churn_model(df):
    le = LabelEncoder()
    X = pd.DataFrame({"Tenure_Months":df["Tenure_Months"],"Satisfaction_Score":df["Satisfaction_Score"],
                       "Avg_Monthly_Hours":df["Avg_Monthly_Hours"],"Last_Evaluation_Score":df["Last_Evaluation_Score"],
                       "Salary_Tier_Num":df["Salary_Tier_Num"],"Dept_Encoded":le.fit_transform(df["Department"])})
    y = df["Churn_Risk"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf = RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_split=5,
                                  class_weight="balanced",random_state=42,n_jobs=-1)
    clf.fit(Xtr,ytr)
    return clf, le, list(X.columns), accuracy_score(yte,clf.predict(Xte))

@st.cache_resource(show_spinner=False)
def build_career_engine():
    vec = TfidfVectorizer(ngram_range=(1,2))
    mat = vec.fit_transform([" ".join(r["skills"]) for r in INTERNAL_ROLES])
    return vec, mat

def recommend_roles(skills_str, top_n=3):
    vec, mat = build_career_engine()
    sims = cosine_similarity(vec.transform([skills_str]), mat).flatten()
    top_idx = sims.argsort()[::-1][:top_n]
    emp_set = set(s.strip() for s in skills_str.split(","))
    results = []
    for idx in top_idx:
        role = INTERNAL_ROLES[idx]; req = set(role["skills"])
        matched = emp_set & req; gap = req - emp_set
        results.append({"title":role["title"],"dept":role["dept"],
                         "similarity":round(float(sims[idx]),4),
                         "match_pct":round(len(matched)/len(req)*100,1),
                         "matched":sorted(matched),"gap":sorted(gap)})
    return results

PL = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
           font=dict(family="Syne,Cairo,sans-serif",color="#7a93b5",size=12),
           colorway=["#2b7fff","#8b5cf6","#00d4ff","#10b981","#f59e0b","#ef4444","#ec4899"],
           margin=dict(l=20,r=20,t=40,b=20))

# ─────────────────────────────────────────────
# CUSTOM SIDEBAR
# ─────────────────────────────────────────────
def render_custom_sidebar():
    page    = st.session_state.get("current_page","hr_summary")
    name    = st.session_state.get("user_name","")
    email   = st.session_state.get("user_email","")
    role    = st.session_state.get("user_role","user")
    is_admin = role == "admin"
    rc = "#2b7fff" if is_admin else "#8b5cf6"
    rl = "🛡️ Admin" if is_admin else "👤 User"

    nav_items = [("hr_summary","📊","HR Executive Summary"),
                 ("risk_predictor","🔮","Flight Risk Predictor"),
                 ("career_mobility","🚀","Career Mobility Engine")]
    if is_admin:
        nav_items.append(("admin_panel","🛡️","Admin Dashboard"))

    nav_html = "".join(
        f'<button class="sb-nbtn {"active" if page==k else ""}" onclick="navTo(\'{k}\')">'
        f'<span class="ico">{ic}</span><span>{lb}</span></button>'
        for k, ic, lb in nav_items
    )

    st.markdown(f"""
    <button id="hBtn" onclick="toggleSB()">
        <span class="hb"></span><span class="hb"></span><span class="hb"></span>
    </button>
    <div id="sbOverlay" onclick="closeSB()"></div>
    <div id="cSidebar">
        <div style="margin-bottom:.4rem;">
            <div class="sb-logo">🧠 AI Talent Engine</div>
            <div class="sb-ver">v2.0 · Enterprise HR Platform</div>
        </div>
        <div class="sb-div"></div>
        <div class="sb-ucard">
            <div class="sb-uname">{name}</div>
            <div class="sb-uemail">{email}</div>
            <span style="display:inline-block;margin-top:.4rem;padding:.1rem .5rem;border-radius:100px;
                         font-size:.65rem;font-family:'JetBrains Mono',monospace;font-weight:600;
                         background:rgba(43,127,255,.15);color:{rc};border:1px solid {rc}55;">{rl}</span>
        </div>
        <div class="sb-nlbl">NAVIGATION</div>
        {nav_html}
        <div style="flex:1;min-height:1rem;"></div>
        <div class="sb-div"></div>
        <button class="sb-lbtn" onclick="doLogout()">
            <span class="ico">🚪</span><span>تسجيل الخروج</span>
        </button>
        <div class="sb-foot">© 2025 AI Talent Engine<br>Powered by Kareem Tamer Temsah</div>
    </div>

    <script>
    (function(){{
        var sb = document.getElementById('cSidebar');
        var ov = document.getElementById('sbOverlay');
        var hb = document.getElementById('hBtn');
        var open = false;

        function openSB() {{
            open=true; sb.classList.add('open'); hb.classList.add('open');
            ov.classList.add('vis');
            requestAnimationFrame(function(){{ ov.classList.add('act'); }});
            document.body.style.overflow='hidden';
        }}
        function closeSBfn() {{
            open=false; sb.classList.remove('open'); hb.classList.remove('open');
            ov.classList.remove('act');
            setTimeout(function(){{ ov.classList.remove('vis'); }}, 300);
            document.body.style.overflow='';
        }}
        window.toggleSB = function(){{ open ? closeSBfn() : openSB(); }};
        window.closeSB  = closeSBfn;

        window.navTo = function(k) {{
            var u = new URL(window.location.href);
            u.searchParams.set('nav', k);
            window.location.href = u.toString();
        }};
        window.doLogout = function() {{
            var u = new URL(window.location.href);
            u.searchParams.set('nav','logout');
            window.location.href = u.toString();
        }};

        document.addEventListener('keydown', function(e){{ if(e.key==='Escape'&&open) closeSBfn(); }});
        window.addEventListener('resize', function(){{ if(window.innerWidth>992&&open) closeSBfn(); }});
    }})();
    </script>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────
def render_login_page():
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                    background:linear-gradient(135deg,#e8f0ff 30%,#00d4ff 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;line-height:1.1;margin-bottom:.5rem;">🧠 AI Talent Engine</div>
        <div style="color:#3d5475;font-family:'JetBrains Mono',monospace;font-size:.7rem;
                    letter-spacing:.15em;text-transform:uppercase;">Enterprise HR Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 تسجيل الدخول","✨ حساب جديد"])
    with tab1:
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        em = st.text_input("البريد الإلكتروني", placeholder="you@company.com", key="le")
        pw = st.text_input("كلمة المرور", type="password", placeholder="••••••••", key="lp")
        if st.button("تسجيل الدخول ←", use_container_width=True, key="bl"):
            if not em or not pw:
                st.markdown('<div class="alert-box alert-error">⚠️ يرجى إدخال البريد وكلمة المرور.</div>', unsafe_allow_html=True)
            else:
                u = login_user(em.strip(), pw)
                if u:
                    st.session_state.update({"authenticated":True,"user_id":u[0],"user_email":u[1],
                                             "user_name":u[2] or u[1],"user_role":u[3],"current_page":"hr_summary"})
                    st.rerun()
                else:
                    st.markdown('<div class="alert-box alert-error">❌ بيانات الدخول غير صحيحة.</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        rn = st.text_input("الاسم الكامل",       placeholder="اسمك الكامل",          key="rn")
        re = st.text_input("البريد الإلكتروني",  placeholder="you@company.com",       key="re")
        rp = st.text_input("كلمة المرور",        type="password", placeholder="6 أحرف على الأقل", key="rp")
        rp2= st.text_input("تأكيد كلمة المرور", type="password", placeholder="أعد الكتابة",       key="rp2")
        if st.button("إنشاء حساب ←", use_container_width=True, key="br"):
            if not all([rn,re,rp,rp2]):
                st.markdown('<div class="alert-box alert-error">⚠️ يرجى ملء جميع الحقول.</div>', unsafe_allow_html=True)
            elif rp != rp2:
                st.markdown('<div class="alert-box alert-error">❌ كلمتا المرور غير متطابقتين.</div>', unsafe_allow_html=True)
            elif len(rp) < 6:
                st.markdown('<div class="alert-box alert-error">❌ كلمة المرور قصيرة جداً.</div>', unsafe_allow_html=True)
            else:
                ok, msg = register_user(re.strip(), rp, rn.strip())
                st.markdown(f'<div class="alert-box {"alert-success" if ok else "alert-error"}">{"✅" if ok else "❌"} {msg}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 1
# ─────────────────────────────────────────────
def page_hr_summary(df):
    st.markdown('<div class="page-hero"><h1>📊 HR Executive Summary</h1><p>نظرة شاملة على صحة القوى العاملة — Workforce Intelligence at a Glance</p></div>', unsafe_allow_html=True)
    total=len(df); avg_sat=df["Satisfaction_Score"].mean(); at_risk=df["Churn_Risk"].sum()
    ret_rate=(1-at_risk/total)*100; avg_eval=df["Last_Evaluation_Score"].mean()
    k1,k2,k3,k4,k5 = st.columns(5)
    for col,ic,v,lb,d,pos in [(k1,"👥",f"{total:,}","إجمالي الموظفين","+12%",True),
                               (k2,"😊",f"{avg_sat:.2f}","متوسط الرضا","+0.03",True),
                               (k3,"🏆",f"{ret_rate:.1f}%","معدل الاحتفاظ","-1.2%",False),
                               (k4,"⚠️",f"{at_risk:,}","موظفون في خطر","+5%",False),
                               (k5,"⭐",f"{avg_eval:.2f}","متوسط التقييم","+0.02",True)]:
        with col:
            st.markdown(f'<div class="stat-card"><span class="stat-icon">{ic}</span><div class="stat-value">{v}</div><div class="stat-label">{lb}</div><div class="stat-delta {"delta-pos" if pos else "delta-neg"}">{d} vs last quarter</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">🏢 توزيع الموظفين حسب القسم</div>', unsafe_allow_html=True)
        dc=df["Department"].value_counts().reset_index(); dc.columns=["Dept","Count"]
        fig=px.bar(dc,x="Count",y="Dept",orientation="h",color="Count",color_continuous_scale=["#1e2d42","#2b7fff"])
        fig.update_traces(marker_line_width=0); fig.update_layout(**PL,height=370,showlegend=False,coloraxis_showscale=False,xaxis=dict(gridcolor="#1e2d42"),yaxis=dict(gridcolor="#1e2d42",tickfont=dict(size=11)))
        st.plotly_chart(fig,use_container_width=True,key="bd")
    with c2:
        st.markdown('<div class="section-title">⚠️ معدل المخاطر حسب القسم</div>', unsafe_allow_html=True)
        rd=df.groupby("Department")["Churn_Risk"].agg(["sum","count"]).reset_index(); rd.columns=["Dept","AR","T"]
        rd["RR"]=(rd["AR"]/rd["T"]*100).round(1); rd=rd.sort_values("RR",ascending=False)
        fig2=px.bar(rd,x="RR",y="Dept",orientation="h",color="RR",color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
        fig2.update_traces(marker_line_width=0); fig2.update_layout(**PL,height=370,showlegend=False,coloraxis_showscale=False,xaxis=dict(gridcolor="#1e2d42"),yaxis=dict(gridcolor="#1e2d42",tickfont=dict(size=11)))
        st.plotly_chart(fig2,use_container_width=True,key="br")
    c3,c4 = st.columns([3,2])
    with c3:
        st.markdown('<div class="section-title">🔗 مصفوفة الارتباط</div>', unsafe_allow_html=True)
        cols=["Tenure_Months","Satisfaction_Score","Avg_Monthly_Hours","Last_Evaluation_Score","Salary_Tier_Num","Churn_Risk"]
        corr=df[cols].corr().round(3); lbls=["مدة الخدمة","الرضا","ساعات العمل","تقييم الأداء","مستوى الراتب","خطر المغادرة"]
        fig3=go.Figure(go.Heatmap(z=corr.values,x=lbls,y=lbls,colorscale=[[0,"#ef4444"],[0.5,"#1e2d42"],[1,"#2b7fff"]],
                                   text=corr.values.round(2),texttemplate="%{text}",textfont=dict(size=10,family="JetBrains Mono"),zmin=-1,zmax=1))
        fig3.update_layout(**PL,height=340); st.plotly_chart(fig3,use_container_width=True,key="hm")
    with c4:
        st.markdown('<div class="section-title">💰 توزيع مستويات الرواتب</div>', unsafe_allow_html=True)
        sc=df["Salary_Tier"].value_counts().reset_index(); sc.columns=["Tier","Count"]
        fig4=px.pie(sc,names="Tier",values="Count",hole=0.55,color_discrete_sequence=["#2b7fff","#8b5cf6","#00d4ff","#10b981","#f59e0b"])
        fig4.update_traces(textfont=dict(family="JetBrains Mono"),textinfo="percent+label")
        fig4.update_layout(**PL,height=340,showlegend=False); st.plotly_chart(fig4,use_container_width=True,key="ps")
    st.markdown('<div class="section-title">📈 الرضا مقابل ساعات العمل (عينة 500)</div>', unsafe_allow_html=True)
    smp=df.sample(500,random_state=1)
    fig5=px.scatter(smp,x="Avg_Monthly_Hours",y="Satisfaction_Score",color="Churn_Risk",
                    color_discrete_map={0:"#10b981",1:"#ef4444"},size="Last_Evaluation_Score",opacity=0.7,
                    hover_data=["Department","Salary_Tier","Tenure_Months"])
    fig5.update_traces(marker=dict(line=dict(width=0)))
    fig5.update_layout(**PL,height=370,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig5,use_container_width=True,key="sc")

# ─────────────────────────────────────────────
# PAGE 2
# ─────────────────────────────────────────────
def page_risk_predictor(df):
    st.markdown('<div class="page-hero"><h1>🔮 Flight Risk Predictor</h1><p>أدخل بيانات الموظف واحصل على تقييم فوري لاحتمالية مغادرته</p></div>', unsafe_allow_html=True)
    with st.spinner("⚙️ تحميل النموذج..."):
        model,le,feat_names,acc = train_churn_model(df)
    st.markdown(f'<div class="alert-box alert-info">🤖 Random Forest (300 شجرة) · دقة: <b>{acc*100:.1f}%</b></div>', unsafe_allow_html=True)
    cf,cr = st.columns([1,1],gap="large")
    with cf:
        st.markdown('<div class="engine-card"><div class="section-title">📋 بيانات الموظف</div>', unsafe_allow_html=True)
        dept  = st.selectbox("القسم",DEPARTMENTS,key="rp_d")
        ten   = st.slider("مدة الخدمة (أشهر)",1,144,24,key="rp_t")
        sat   = st.slider("درجة الرضا",0.1,1.0,0.65,0.01,key="rp_s")
        hrs   = st.slider("ساعات العمل الشهرية",120,310,170,key="rp_h")
        evl   = st.slider("درجة التقييم",0.2,1.0,0.70,0.01,key="rp_e")
        sal   = st.selectbox("مستوى الراتب",SALARY_TIERS,index=1,key="rp_sl")
        btn   = st.button("🔮 تحليل احتمالية المغادرة",use_container_width=True,key="bp")
        st.markdown('</div>', unsafe_allow_html=True)
    with cr:
        if btn:
            sn = SALARY_TIERS.index(sal)+1
            try: de = le.transform([dept])[0]
            except: de = 0
            Xi = pd.DataFrame([[ten,sat,hrs,evl,sn,de]],columns=feat_names)
            prob = model.predict_proba(Xi)[0][1]; pct = round(prob*100,1)
            if pct>=70:   lvl,cls,gc="مرتفع جداً","risk-high","#ef4444"
            elif pct>=45: lvl,cls,gc="متوسط","risk-medium","#f59e0b"
            else:         lvl,cls,gc="منخفض","risk-low","#10b981"
            fig_g=go.Figure(go.Indicator(mode="gauge+number+delta",value=pct,
                number={"suffix":"%","font":{"family":"JetBrains Mono","size":38,"color":gc}},
                delta={"reference":30,"increasing":{"color":"#ef4444"},"decreasing":{"color":"#10b981"}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#3d5475","tickfont":{"family":"JetBrains Mono","size":9}},
                       "bar":{"color":gc,"thickness":0.22},"bgcolor":"#0e1520","borderwidth":0,
                       "steps":[{"range":[0,40],"color":"rgba(16,185,129,0.10)"},{"range":[40,70],"color":"rgba(245,158,11,0.10)"},{"range":[70,100],"color":"rgba(239,68,68,0.10)"}],
                       "threshold":{"line":{"color":gc,"width":3},"value":pct}},
                title={"text":"احتمالية المغادرة","font":{"family":"Syne","size":13,"color":"#7a93b5"}}))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=280,margin=dict(l=20,r=20,t=30,b=10))
            st.plotly_chart(fig_g,use_container_width=True,key="gauge")
            st.markdown(f'<div class="engine-card" style="text-align:center;margin-top:-.5rem;"><div style="font-size:.8rem;color:var(--text-secondary);">مستوى الخطر</div><div class="{cls}" style="font-size:1.5rem;margin:.35rem 0;">{lvl}</div><div style="font-size:.72rem;color:var(--text-muted);font-family:JetBrains Mono,monospace;">Confidence: {max(prob,1-prob)*100:.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="margin-top:1rem;">🔍 أهم العوامل</div>', unsafe_allow_html=True)
            nmap={"Tenure_Months":"مدة الخدمة","Satisfaction_Score":"الرضا","Avg_Monthly_Hours":"ساعات العمل","Last_Evaluation_Score":"تقييم الأداء","Salary_Tier_Num":"مستوى الراتب","Dept_Encoded":"القسم"}
            fd=pd.DataFrame({"Feature":[nmap.get(f,f) for f in feat_names],"Importance":model.feature_importances_}).sort_values("Importance",ascending=True)
            fig_i=px.bar(fd,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=["#1e2d42","#2b7fff","#00d4ff"])
            fig_i.update_traces(marker_line_width=0); fig_i.update_layout(**PL,height=245,showlegend=False,coloraxis_showscale=False)
            st.plotly_chart(fig_i,use_container_width=True,key="fi")
        else:
            st.markdown('<div class="engine-card" style="text-align:center;padding:3rem 2rem;opacity:.55;"><div style="font-size:3rem;margin-bottom:1rem;">🔮</div><div style="font-family:Syne,sans-serif;color:var(--text-secondary);">أدخل بيانات الموظف واضغط تحليل</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 3
# ─────────────────────────────────────────────
def page_career_mobility(df):
    st.markdown('<div class="page-hero"><h1>🚀 Smart Career Mobility Engine</h1><p>اكتشف أفضل المسارات الداخلية بناءً على مهاراتك</p></div>', unsafe_allow_html=True)
    build_career_engine()
    cl,cr = st.columns([1,1.4],gap="large")
    with cl:
        st.markdown('<div class="engine-card"><div class="section-title">👤 ملف الموظف</div>', unsafe_allow_html=True)
        mode = st.radio("طريقة الإدخال",["اختر موظفاً موجوداً","أدخل مهاراتك يدوياً"],horizontal=True,key="cm_m")
        if mode=="اختر موظفاً موجوداً":
            sd=st.selectbox("القسم",DEPARTMENTS,key="cm_d")
            ddf=df[df["Department"]==sd].sample(min(20,len(df[df["Department"]==sd])),random_state=7)
            se=st.selectbox("اختر الموظف",ddf["Employee_ID"].tolist(),key="cm_e")
            emp=df[df["Employee_ID"]==se].iloc[0]; skills_input=emp["Skills"]
            st.markdown(f'<div style="margin-top:.8rem;"><div style="font-size:.75rem;color:var(--text-secondary);margin-bottom:.35rem;font-weight:600;">المهارات</div>{"".join(f\'<span class="skill-tag">{s.strip()}</span>\' for s in skills_input.split(","))}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="margin-top:.7rem;display:grid;grid-template-columns:1fr 1fr;gap:.35rem;"><div style="background:var(--bg-base);border-radius:6px;padding:.5rem .7rem;"><div style="font-size:.62rem;color:var(--text-muted);font-family:JetBrains Mono,monospace;">SATISFACTION</div><div style="font-size:.9rem;color:var(--accent-cyan);font-weight:700;font-family:JetBrains Mono,monospace;">{emp["Satisfaction_Score"]:.2f}</div></div><div style="background:var(--bg-base);border-radius:6px;padding:.5rem .7rem;"><div style="font-size:.62rem;color:var(--text-muted);font-family:JetBrains Mono,monospace;">TENURE</div><div style="font-size:.9rem;color:var(--accent-cyan);font-weight:700;font-family:JetBrains Mono,monospace;">{emp["Tenure_Months"]} mos</div></div><div style="background:var(--bg-base);border-radius:6px;padding:.5rem .7rem;"><div style="font-size:.62rem;color:var(--text-muted);font-family:JetBrains Mono,monospace;">EVALUATION</div><div style="font-size:.9rem;color:var(--accent-green);font-weight:700;font-family:JetBrains Mono,monospace;">{emp["Last_Evaluation_Score"]:.2f}</div></div><div style="background:var(--bg-base);border-radius:6px;padding:.5rem .7rem;"><div style="font-size:.62rem;color:var(--text-muted);font-family:JetBrains Mono,monospace;">CHURN RISK</div><div style="font-size:.9rem;font-weight:700;font-family:JetBrains Mono,monospace;color:{"#ef4444" if emp["Churn_Risk"]==1 else "#10b981";">{"HIGH ⚠️" if emp["Churn_Risk"]==1 else "LOW ✓"}</div></div></div>', unsafe_allow_html=True)
        else:
            pool_all=sorted(set(s for p in SKILLS_POOL.values() for s in p))
            chosen=st.multiselect("اختر مهاراتك",pool_all,default=["Python","SQL","Machine Learning"],key="cm_cs")
            skills_input=", ".join(chosen) if chosen else ""
        ab=st.button("🚀 تحليل المسار الوظيفي",use_container_width=True,key="bca")
        st.markdown('</div>', unsafe_allow_html=True)
    with cr:
        if ab and skills_input:
            results=recommend_roles(skills_input,top_n=3)
            st.markdown('<div class="section-title">🎯 أفضل 3 وظائف داخلية مقترحة</div>', unsafe_allow_html=True)
            medals=["🥇","🥈","🥉"]; colors=["#2b7fff","#8b5cf6","#10b981"]
            for i,role in enumerate(results):
                mc=colors[i%len(colors)]
                matched_html="".join(f'<span class="skill-tag">{s}</span>' for s in role["matched"])
                gap_html="".join(f'<span class="skill-gap-tag">{s}</span>' for s in role["gap"])
                st.markdown(f"""
                <div class="job-card">
                    <div style="display:flex;justify-content:space-between;align-items:start;">
                        <div><span style="font-size:1rem;margin-right:.3rem;">{medals[i]}</span>
                            <h4 style="display:inline;color:{mc} !important;">{role["title"]}</h4>
                            <div style="margin-top:.22rem;"><span class="badge badge-blue">{role["dept"]}</span></div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-family:JetBrains Mono,monospace;font-size:1.3rem;font-weight:700;color:{mc};">{role["match_pct"]}%</div>
                            <div style="font-size:.62rem;color:var(--text-muted);font-family:JetBrains Mono,monospace;">تطابق المهارات</div>
                        </div>
                    </div>
                    <div class="progress-bar-wrap" style="margin:.65rem 0 .5rem;">
                        <div class="progress-bar-fill" style="width:{role["match_pct"]}%;background:linear-gradient(90deg,{mc}99,{mc});"></div>
                    </div>
                    <div style="font-size:.68rem;color:var(--text-muted);font-family:JetBrains Mono,monospace;margin-bottom:.45rem;">Cosine Similarity: {round(role["similarity"]*100,1)}%</div>
                    {f'<div style="margin-bottom:.4rem;"><div style="font-size:.68rem;color:var(--accent-green);font-weight:600;margin-bottom:.2rem;">✅ مهاراتك المطابقة</div>{matched_html}</div>' if role["matched"] else ""}
                    {f'<div><div style="font-size:.68rem;color:var(--accent-orange);font-weight:600;margin-bottom:.2rem;">📚 فجوة المهارات</div>{gap_html}</div>' if role["gap"] else ""}
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 4
# ─────────────────────────────────────────────
def page_admin(df):
    st.markdown('<div class="page-hero" style="border-color:#2b7fff44;"><h1>🛡️ Admin Control Center</h1><p>إحصائيات النظام الشاملة وإدارة المستخدمين</p></div>', unsafe_allow_html=True)
    udf=get_all_users(); tu=len(udf); ac=(udf["role"]=="admin").sum()
    s1,s2,s3,s4=st.columns(4)
    for col,ic,v,lb in [(s1,"👥",str(tu),"إجمالي المستخدمين"),(s2,"🛡️",str(ac),"المسؤولون"),(s3,"👤",str(tu-ac),"المستخدمون العاديون"),(s4,"💾",f"{len(df):,}","سجلات الموظفين")]:
        with col:
            st.markdown(f'<div class="stat-card"><span class="stat-icon">{ic}</span><div class="stat-value">{v}</div><div class="stat-label">{lb}</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">👥 إدارة المستخدمين</div>', unsafe_allow_html=True)
    cs,co=st.columns([2,1])
    with cs:
        uid=st.selectbox("اختر المستخدم",udf["id"].tolist(),
                          format_func=lambda x:f"{udf[udf['id']==x]['email'].iloc[0]} ({udf[udf['id']==x]['role'].iloc[0]})")
    sr=udf[udf["id"]==uid]["role"].iloc[0]
    with co:
        if st.button("🗑️ حذف",key="du"):
            ok,msg=delete_user(uid,st.session_state["user_id"])
            (st.success if ok else st.error)(msg)
            if ok: st.rerun()
        if sr!="admin":
            if st.button("⭐ ترقية إلى Admin",key="pu"):
                ok,msg=promote_to_admin(uid)
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()
        else:
            if st.button("⬇️ خفض إلى User",key="dmu"):
                ok,msg=demote_to_user(uid,st.session_state["user_id"],st.session_state["user_role"])
                (st.success if ok else st.error)(msg)
                if ok: st.rerun()
    st.markdown("---")
    disp=udf.copy(); disp.columns=["ID","البريد","الاسم","الدور","تاريخ الإنشاء","آخر دخول"]
    st.dataframe(disp,use_container_width=True,height=280)
    st.markdown("---")
    st.markdown('<div class="section-title">📊 إحصائيات</div>', unsafe_allow_html=True)
    ca,cb=st.columns(2)
    with ca:
        ds=df.groupby(["Department","Salary_Tier"]).size().reset_index(name="Count")
        pv=ds.pivot(index="Department",columns="Salary_Tier",values="Count").fillna(0)
        fh=go.Figure(go.Heatmap(z=pv.values,x=pv.columns.tolist(),y=pv.index.tolist(),
                                  colorscale=[[0,"#0e1520"],[1,"#2b7fff"]],texttemplate="%{z:.0f}",textfont=dict(size=10,family="JetBrains Mono")))
        fh.update_layout(**PL,title="الأقسام × مستوى الراتب",height=400)
        st.plotly_chart(fh,use_container_width=True,key="ah")
    with cb:
        model,_,_,acc=train_churn_model(df)
        fp=px.pie(names=["مدة الخدمة","الرضا","ساعات العمل","تقييم الأداء","مستوى الراتب","القسم"],
                   values=model.feature_importances_,hole=0.4,
                   color_discrete_sequence=["#2b7fff","#8b5cf6","#00d4ff","#10b981","#f59e0b","#ef4444"],
                   title=f"أهمية المتغيرات — دقة: {acc*100:.1f}%")
        fp.update_layout(**PL,height=400); st.plotly_chart(fp,use_container_width=True,key="ap")
    st.markdown('<div class="section-title">🗄️ معاينة قاعدة البيانات</div>', unsafe_allow_html=True)
    f1,f2,f3=st.columns(3)
    with f1: fd=st.selectbox("القسم",["الكل"]+DEPARTMENTS,key="af1")
    with f2: fr=st.selectbox("الخطر",["الكل","مرتفع","منخفض"],key="af2")
    with f3: fs=st.selectbox("الراتب",["الكل"]+SALARY_TIERS,key="af3")
    flt=df.copy()
    if fd!="الكل":   flt=flt[flt["Department"]==fd]
    if fr=="مرتفع": flt=flt[flt["Churn_Risk"]==1]
    if fr=="منخفض": flt=flt[flt["Churn_Risk"]==0]
    if fs!="الكل":   flt=flt[flt["Salary_Tier"]==fs]
    sc=["Employee_ID","Department","Tenure_Months","Satisfaction_Score","Avg_Monthly_Hours","Last_Evaluation_Score","Salary_Tier","Churn_Risk"]
    st.dataframe(flt[sc].head(100),use_container_width=True,height=360)
    st.markdown(f'<div style="color:var(--text-muted);font-size:.72rem;font-family:JetBrains Mono,monospace;">عرض 100 من {len(flt):,}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    init_db()

    # معالجة query params للـ navigation
    params = st.query_params
    if "nav" in params:
        nav_val = params["nav"]
        if nav_val == "logout":
            for k in ["authenticated","user_id","user_email","user_name","user_role","current_page"]:
                st.session_state.pop(k,None)
            st.query_params.clear(); st.rerun()
        elif nav_val in ["hr_summary","risk_predictor","career_mobility","admin_panel"]:
            st.session_state["current_page"] = nav_val
            st.query_params.clear(); st.rerun()

    if not st.session_state.get("authenticated"):
        _,center,_ = st.columns([1,2,1])
        with center: render_login_page()
        return

    with st.spinner("🔄 تحميل البيانات..."):
        df = generate_employee_data(2000)

    render_custom_sidebar()

    page = st.session_state.get("current_page","hr_summary")
    try:
        if   page=="hr_summary":     page_hr_summary(df)
        elif page=="risk_predictor": page_risk_predictor(df)
        elif page=="career_mobility":page_career_mobility(df)
        elif page=="admin_panel":
            if st.session_state.get("user_role")=="admin": page_admin(df)
            else: st.error("⛔ غير مصرح. هذه الصفحة للمسؤولين فقط.")
        else: page_hr_summary(df)
    except Exception as e:
        st.error(f"❌ حدث خطأ: {e}"); st.exception(e)

if __name__=="__main__":
    main()
