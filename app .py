"""
╔══════════════════════════════════════════════════════════════════╗
║   AI Talent Retention & Career Mobility Engine                   ║
║   محرك الذكاء الاصطناعي للاحتفاظ بالمواهب والمسار الوظيفي     ║
║   Version: 2.0  |  Built with Streamlit + ML + NLP              ║
║   Responsive + Animated Navbar + Admin User Management          ║
╚══════════════════════════════════════════════════════════════════╝
"""

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

# ─────────────────────────────────────────────
# ML & NLP Imports
# ─────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Talent Engine | محرك المواهب",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Enterprise Dark Mode + Responsive + Animated Navbar
# ─────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Cairo:wght@300;400;600;700&display=swap');

:root {
    --bg-base:       #080c14;
    --bg-card:       #0e1520;
    --bg-card2:      #111a28;
    --border:        #1e2d42;
    --border-glow:   #1e4a8a;
    --accent-blue:   #2b7fff;
    --accent-cyan:   #00d4ff;
    --accent-purple: #8b5cf6;
    --accent-green:  #10b981;
    --accent-orange: #f59e0b;
    --accent-red:    #ef4444;
    --text-primary:  #e8f0ff;
    --text-secondary:#7a93b5;
    --text-muted:    #3d5475;
    --gradient-hero: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #091427 100%);
    --gradient-card: linear-gradient(145deg, #0e1520 0%, #111d30 100%);
    --gradient-accent: linear-gradient(135deg, #2b7fff 0%, #8b5cf6 100%);
    --shadow-card:   0 4px 24px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.03) inset;
    --shadow-glow:   0 0 40px rgba(43,127,255,0.15);
    --radius:        14px;
    --radius-sm:     8px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Cairo', 'Syne', sans-serif !important;
}

/* ========== SIDEBAR SLIDER (محسن) ========== */
[data-testid="stSidebar"] {
    background: #060b12 !important;
    border-right: 1px solid var(--border) !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    z-index: 1001 !important;
    will-change: transform;
}

@media (min-width: 993px) {
    [data-testid="stSidebar"] {
        transform: translateX(0) !important;
    }
}

@media (max-width: 992px) {
    [data-testid="stSidebar"] {
        transform: translateX(-100%);
        position: fixed !important;
        top: 0;
        left: 0;
        height: 100vh !important;
        width: 280px !important;
        box-shadow: 4px 0 20px rgba(0,0,0,0.5);
    }
    [data-testid="stSidebar"][data-state="expanded"] {
        transform: translateX(0) !important;
    }
    .block-container {
        padding: 1rem !important;
        padding-top: 70px !important;
    }
    .page-hero h1 { font-size: 1.4rem !important; }
    .page-hero p { max-width: 100% !important; font-size: 0.85rem !important; }
    .stat-card .stat-value { font-size: 1.6rem !important; }
    .stat-card .stat-label { font-size: 0.7rem !important; }
    .section-title { font-size: 1rem !important; }
    .job-card h4 { font-size: 0.9rem !important; }
    .stDataFrame {
        overflow-x: auto !important;
    }
    .stPlotlyChart {
        height: 280px !important;
    }
}

.hamburger-btn {
    display: none;
    position: fixed;
    top: 16px;
    left: 16px;
    z-index: 1100;
    background: var(--gradient-accent);
    border: none;
    border-radius: 10px;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 1.5rem;
    color: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    transition: all 0.2s;
}
.hamburger-btn:hover {
    transform: scale(1.02);
    background: #2b7fff;
}
@media (max-width: 992px) {
    .hamburger-btn {
        display: block;
    }
}

.sidebar-overlay {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.6);
    z-index: 1000;
    backdrop-filter: blur(3px);
    transition: opacity 0.2s;
}
.sidebar-overlay.active {
    display: block;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 3px; }

#MainMenu, footer, header { visibility: hidden !important; }
</style>
"""

CARD_CSS = """
<style>
.engine-card {
    background: var(--gradient-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-card);
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.engine-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-accent);
    opacity: 0.7;
}
.engine-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s;
}
.stat-card:hover { transform: translateY(-3px); border-color: var(--border-glow); box-shadow: var(--shadow-glow); }
.stat-card .stat-icon { font-size: 2rem; margin-bottom: 0.5rem; display: block; }
.stat-card .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.stat-card .stat-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
}
.stat-card .stat-delta {
    font-size: 0.75rem;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.stat-card .delta-pos { color: var(--accent-green); }
.stat-card .delta-neg { color: var(--accent-red); }

.page-hero {
    background: var(--gradient-hero);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.page-hero::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(43,127,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.page-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8f0ff 30%, var(--accent-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.page-hero p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    max-width: 60%;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 100px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-blue   { background: rgba(43,127,255,0.15); color: var(--accent-blue); border: 1px solid rgba(43,127,255,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--accent-green); border: 1px solid rgba(16,185,129,0.3); }
.badge-red    { background: rgba(239,68,68,0.15);  color: var(--accent-red);   border: 1px solid rgba(239,68,68,0.3); }
.badge-orange { background: rgba(245,158,11,0.15); color: var(--accent-orange);border: 1px solid rgba(245,158,11,0.3); }
.badge-purple { background: rgba(139,92,246,0.15); color: var(--accent-purple);border: 1px solid rgba(139,92,246,0.3); }

.skill-tag {
    display: inline-block;
    background: rgba(43,127,255,0.1);
    border: 1px solid rgba(43,127,255,0.25);
    color: var(--accent-cyan);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
}
.skill-gap-tag {
    display: inline-block;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    color: var(--accent-orange);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, var(--border), transparent);
}

.risk-high   { color: var(--accent-red);    font-weight: 700; }
.risk-medium { color: var(--accent-orange); font-weight: 700; }
.risk-low    { color: var(--accent-green);  font-weight: 700; }

.progress-bar-wrap {
    background: var(--border);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.3rem;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: var(--gradient-accent);
    transition: width 0.6s ease;
}

.job-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: all 0.2s;
}
.job-card:hover { border-color: var(--accent-blue); }
.job-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent-cyan);
    margin-bottom: 0.3rem;
}

.login-container {
    max-width: 440px;
    margin: 3rem auto;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: var(--shadow-card), 0 0 80px rgba(43,127,255,0.08);
    position: relative;
    overflow: hidden;
}
.login-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-accent);
}

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sidebar-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}

.alert-box {
    padding: 0.8rem 1.2rem;
    border-radius: var(--radius-sm);
    font-family: 'Cairo', sans-serif;
    font-size: 0.88rem;
    margin-bottom: 1rem;
}
.alert-error   { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.3);  color: #fca5a5; }
.alert-success { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); color: #6ee7b7; }
.alert-info    { background: rgba(43,127,255,0.1); border: 1px solid rgba(43,127,255,0.3); color: #93c5fd; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HAMBURGER BUTTON + OVERLAY (JavaScript for mobile sidebar toggle)
# ─────────────────────────────────────────────
st.markdown("""
<button class="hamburger-btn" id="hamburgerBtn">☰</button>
<div class="sidebar-overlay" id="sidebarOverlay"></div>

<script>
(function() {
    let sidebar = null;
    let hamburger = document.getElementById('hamburgerBtn');
    let overlay = document.getElementById('sidebarOverlay');

    function getSidebar() {
        if (!sidebar) {
            sidebar = document.querySelector('[data-testid="stSidebar"]');
        }
        return sidebar;
    }

    function toggleSidebar(forceExpand = null) {
        const sb = getSidebar();
        if (!sb) return;
        const isExpanded = sb.getAttribute('data-state') === 'expanded';
        let newState = forceExpand !== null ? forceExpand : !isExpanded;
        if (newState) {
            sb.setAttribute('data-state', 'expanded');
            if (overlay) overlay.classList.add('active');
        } else {
            sb.setAttribute('data-state', 'collapsed');
            if (overlay) overlay.classList.remove('active');
        }
    }

    function closeSidebar() {
        toggleSidebar(false);
    }

    if (hamburger) {
        hamburger.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleSidebar();
        });
    }
    if (overlay) {
        overlay.addEventListener('click', closeSidebar);
    }

    function addCloseOnNavLinks() {
        const sb = getSidebar();
        if (sb) {
            const links = sb.querySelectorAll('button');
            links.forEach(btn => {
                btn.removeEventListener('click', closeSidebar);
                btn.addEventListener('click', closeSidebar);
            });
        }
    }

    const observer = new MutationObserver(addCloseOnNavLinks);
    function observeSidebar() {
        const sb = getSidebar();
        if (sb) {
            observer.observe(sb, { childList: true, subtree: true });
            addCloseOnNavLinks();
        }
    }
    setTimeout(observeSidebar, 500);

    function handleResize() {
        const sb = getSidebar();
        if (!sb) return;
        if (window.innerWidth > 992) {
            sb.setAttribute('data-state', 'expanded');
            if (overlay) overlay.classList.remove('active');
        } else {
            sb.setAttribute('data-state', 'collapsed');
            if (overlay) overlay.classList.remove('active');
        }
    }
    window.addEventListener('resize', handleResize);
    handleResize();
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE — SQLite Auth
# ─────────────────────────────────────────────
DB_PATH = "talent_engine.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'user',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT
        )
    """)
    admin_email = "kareemeltemsah7@gmail.com"
    admin_pw    = hash_password("temsah1")
    c.execute("""
        INSERT OR IGNORE INTO users (email, password_hash, full_name, role)
        VALUES (?, ?, ?, ?)
    """, (admin_email, admin_pw, "Kareem El-Temsah", "admin"))
    conn.commit()
    conn.close()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def login_user(email: str, password: str):
    conn = get_db()
    c = conn.cursor()
    ph = hash_password(password)
    c.execute("SELECT id, email, full_name, role FROM users WHERE email=? AND password_hash=?", (email, ph))
    row = c.fetchone()
    if row:
        c.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.now().isoformat(), row[0]))
        conn.commit()
    conn.close()
    return row

def register_user(email: str, password: str, full_name: str):
    conn = get_db()
    c = conn.cursor()
    try:
        ph = hash_password(password)
        c.execute("INSERT INTO users (email, password_hash, full_name, role) VALUES (?,?,?,?)",
                  (email, ph, full_name, "user"))
        conn.commit()
        conn.close()
        return True, "تم إنشاء الحساب بنجاح!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "البريد الإلكتروني مسجل مسبقاً."
    except Exception as e:
        conn.close()
        return False, str(e)

def get_all_users():
    conn = get_db()
    df = pd.read_sql("SELECT id, email, full_name, role, created_at, last_login FROM users", conn)
    conn.close()
    return df

# دوال إدارة المستخدمين الجديدة
def delete_user(user_id: int, current_user_id: int):
    """حذف مستخدم مع منع حذف المستخدم الحالي"""
    if user_id == current_user_id:
        return False, "لا يمكنك حذف حسابك الحالي."
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted, "تم حذف المستخدم." if deleted else "المستخدم غير موجود."

def promote_to_admin(user_id: int):
    """ترقية مستخدم إلى Admin"""
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE users SET role = 'admin' WHERE id = ?", (user_id,))
    updated = c.rowcount > 0
    conn.commit()
    conn.close()
    return updated, "تمت الترقية إلى Admin."

def demote_to_user(user_id: int, current_user_id: int, current_user_role: str):
    """خفض Admin إلى user، مع منع خفض آخر Admin"""
    if user_id == current_user_id and current_user_role == "admin":
        return False, "لا يمكنك خفض دورك بنفسك."
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    admin_count = c.fetchone()[0]
    if admin_count <= 1:
        return False, "يجب أن يبقى مدير واحد على الأقل في النظام."
    c.execute("UPDATE users SET role = 'user' WHERE id = ?", (user_id,))
    updated = c.rowcount > 0
    conn.commit()
    conn.close()
    return updated, "تم خفض الدور إلى User."

# ─────────────────────────────────────────────
# DATA GENERATOR — 2000 Realistic Employees
# ─────────────────────────────────────────────
DEPARTMENTS = ["Engineering", "Product", "Data Science", "Design", "Sales",
               "Marketing", "Finance", "HR", "Operations", "Customer Success",
               "Legal", "Security", "DevOps", "QA", "Research"]

SALARY_TIERS = ["Tier 1 (Low)", "Tier 2 (Mid)", "Tier 3 (Senior)", "Tier 4 (Lead)", "Tier 5 (Executive)"]

SKILLS_POOL = {
    "Engineering":       ["Python", "Java", "Go", "Kubernetes", "AWS", "React", "TypeScript", "PostgreSQL", "Redis", "gRPC", "Docker", "Terraform", "CI/CD", "GraphQL"],
    "Product":           ["Roadmapping", "PRD Writing", "A/B Testing", "User Research", "JIRA", "Analytics", "OKRs", "Stakeholder Management", "SQL", "Figma"],
    "Data Science":      ["Python", "R", "Machine Learning", "Deep Learning", "SQL", "Spark", "Tableau", "Statistics", "NLP", "Computer Vision", "MLflow", "Pandas"],
    "Design":            ["Figma", "UI Design", "UX Research", "Prototyping", "Motion Design", "Brand Identity", "Accessibility", "Design Systems", "Sketch"],
    "Sales":             ["CRM", "Salesforce", "Lead Generation", "Negotiation", "Account Management", "Forecasting", "Cold Outreach", "SaaS Sales", "Demos"],
    "Marketing":         ["SEO", "Content Strategy", "Paid Media", "Email Marketing", "Analytics", "HubSpot", "Copywriting", "Brand Management", "Social Media"],
    "Finance":           ["Financial Modeling", "Excel", "Python", "Risk Analysis", "Budgeting", "FP&A", "GAAP", "Bloomberg", "QuickBooks", "SQL"],
    "HR":                ["Recruiting", "HRIS", "Compensation Design", "L&D", "Performance Management", "Employment Law", "ATS", "Onboarding", "Culture"],
    "Operations":        ["Process Optimization", "Six Sigma", "Supply Chain", "ERP", "Vendor Management", "Data Analysis", "Lean", "Project Management"],
    "Customer Success":  ["Onboarding", "Churn Prevention", "CRM", "Zendesk", "SLAs", "NPS", "Upselling", "Product Adoption", "Communication"],
    "Legal":             ["Contract Review", "IP Law", "Privacy (GDPR)", "Employment Law", "M&A", "Compliance", "Regulatory Affairs"],
    "Security":          ["Penetration Testing", "SIEM", "Zero Trust", "SOC", "Cloud Security", "Python", "Incident Response", "Compliance", "Threat Modeling"],
    "DevOps":            ["Kubernetes", "Terraform", "AWS", "Ansible", "CI/CD", "Docker", "Monitoring", "Linux", "Python", "Infrastructure as Code"],
    "QA":                ["Test Automation", "Selenium", "Pytest", "Manual Testing", "Performance Testing", "API Testing", "JIRA", "Bug Reporting"],
    "Research":          ["Research Design", "Statistics", "Python", "R", "Literature Review", "Grant Writing", "LaTeX", "NLP", "Academic Publishing"],
}

INTERNAL_ROLES = [
    {"title": "Senior ML Engineer",         "dept": "Data Science", "skills": ["Python", "Machine Learning", "Deep Learning", "MLflow", "AWS", "SQL", "Statistics"]},
    {"title": "Product Manager II",          "dept": "Product",      "skills": ["Roadmapping", "A/B Testing", "SQL", "OKRs", "Analytics", "Stakeholder Management"]},
    {"title": "DevOps Lead",                 "dept": "DevOps",       "skills": ["Kubernetes", "Terraform", "AWS", "Docker", "CI/CD", "Python", "Monitoring"]},
    {"title": "UX Research Lead",            "dept": "Design",       "skills": ["UX Research", "Figma", "User Research", "Prototyping", "Design Systems"]},
    {"title": "Data Engineer",               "dept": "Data Science", "skills": ["Python", "Spark", "SQL", "Pandas", "AWS", "Airflow", "PostgreSQL"]},
    {"title": "Head of Customer Success",    "dept": "Customer Success", "skills": ["Churn Prevention", "NPS", "CRM", "Upselling", "SLAs", "Onboarding"]},
    {"title": "Security Architect",          "dept": "Security",     "skills": ["Zero Trust", "Cloud Security", "Threat Modeling", "SIEM", "Python", "Compliance"]},
    {"title": "Frontend Engineer III",       "dept": "Engineering",  "skills": ["React", "TypeScript", "GraphQL", "CSS", "Testing", "Figma", "Performance"]},
    {"title": "Growth Marketing Manager",    "dept": "Marketing",    "skills": ["Paid Media", "Analytics", "A/B Testing", "SEO", "Content Strategy", "HubSpot"]},
    {"title": "Finance Business Partner",    "dept": "Finance",      "skills": ["Financial Modeling", "FP&A", "Python", "SQL", "Excel", "Budgeting", "Analytics"]},
    {"title": "Platform Engineer",           "dept": "DevOps",       "skills": ["Kubernetes", "Terraform", "Go", "AWS", "CI/CD", "gRPC", "Infrastructure as Code"]},
    {"title": "Research Scientist",          "dept": "Research",     "skills": ["Python", "Deep Learning", "NLP", "Statistics", "PyTorch", "Research Design", "LaTeX"]},
    {"title": "Enterprise Account Executive","dept": "Sales",        "skills": ["SaaS Sales", "CRM", "Salesforce", "Negotiation", "Forecasting", "Account Management"]},
    {"title": "HR Business Partner",         "dept": "HR",           "skills": ["Performance Management", "L&D", "Compensation Design", "Recruiting", "Culture", "HRIS"]},
    {"title": "Backend Engineer Staff",      "dept": "Engineering",  "skills": ["Go", "Python", "PostgreSQL", "Redis", "gRPC", "AWS", "Kubernetes", "Architecture"]},
]

@st.cache_data(show_spinner=False)
def generate_employee_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    depts = rng.choice(DEPARTMENTS, size=n, p=None)

    tenure = rng.integers(1, 145, size=n).astype(float)
    tenure_norm = (tenure - tenure.min()) / (tenure.max() - tenure.min())

    salary_probs_base = rng.dirichlet(np.ones(5), size=n)
    for i in range(n):
        t = tenure_norm[i]
        salary_probs_base[i, 0] = max(0.05, salary_probs_base[i, 0] * (1 - t))
        salary_probs_base[i, 4] = salary_probs_base[i, 4] * (0.5 + t)
    salary_probs_base /= salary_probs_base.sum(axis=1, keepdims=True)
    salary_idx = np.array([rng.choice(5, p=row) for row in salary_probs_base])
    salary_tier = [SALARY_TIERS[i] for i in salary_idx]

    base_hours = 140 + rng.normal(0, 20, size=n)
    overwork_bonus = (4 - salary_idx) * rng.uniform(0, 8, size=n)
    avg_hours = np.clip(base_hours + overwork_bonus, 120, 310).round(1)

    raw_sat = (
        0.5 - 0.0015 * (avg_hours - 160)
        + 0.05 * salary_idx
        + 0.002 * tenure_norm * 10
        + rng.normal(0, 0.12, size=n)
    )
    satisfaction = np.clip(raw_sat, 0.1, 1.0).round(3)

    evaluation = np.clip(
        rng.normal(0.68, 0.12, size=n) + 0.04 * salary_idx / 4,
        0.2, 1.0
    ).round(3)

    skills_list = []
    for dept in depts:
        pool = SKILLS_POOL.get(dept, SKILLS_POOL["Engineering"])
        k = rng.integers(3, min(len(pool) + 1, 8))
        chosen = rng.choice(pool, size=k, replace=False).tolist()
        skills_list.append(", ".join(chosen))

    churn_score = (
        0.30 * (1 - satisfaction)
        + 0.25 * (avg_hours - 140) / 170
        + 0.15 * (1 - evaluation)
        + 0.20 * (1 - salary_idx / 4)
        + 0.10 * (1 - tenure_norm)
        + rng.normal(0, 0.05, size=n)
    )
    churn_risk = (churn_score > np.percentile(churn_score, 70)).astype(int)

    df = pd.DataFrame({
        "Employee_ID":          [f"EMP-{10000 + i}" for i in range(n)],
        "Department":           depts,
        "Tenure_Months":        tenure.astype(int),
        "Satisfaction_Score":   satisfaction,
        "Avg_Monthly_Hours":    avg_hours,
        "Last_Evaluation_Score":evaluation,
        "Salary_Tier":          salary_tier,
        "Salary_Tier_Num":      salary_idx + 1,
        "Skills":               skills_list,
        "Churn_Risk":           churn_risk,
        "Churn_Score_Raw":      np.clip(churn_score, 0, 1).round(4),
    })
    return df

# ─────────────────────────────────────────────
# ML MODEL — Random Forest Churn Predictor
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_churn_model(df: pd.DataFrame):
    le_dept = LabelEncoder()
    X = pd.DataFrame({
        "Tenure_Months":        df["Tenure_Months"],
        "Satisfaction_Score":   df["Satisfaction_Score"],
        "Avg_Monthly_Hours":    df["Avg_Monthly_Hours"],
        "Last_Evaluation_Score":df["Last_Evaluation_Score"],
        "Salary_Tier_Num":      df["Salary_Tier_Num"],
        "Dept_Encoded":         le_dept.fit_transform(df["Department"]),
    })
    y = df["Churn_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    return clf, le_dept, list(X.columns), acc

# ─────────────────────────────────────────────
# NLP — Career Mobility via Cosine Similarity
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_career_engine():
    role_texts = [" ".join(r["skills"]) for r in INTERNAL_ROLES]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    role_matrix = vectorizer.fit_transform(role_texts)
    return vectorizer, role_matrix

def recommend_roles(employee_skills_str: str, top_n: int = 3):
    vectorizer, role_matrix = build_career_engine()
    emp_vec = vectorizer.transform([employee_skills_str])
    sims = cosine_similarity(emp_vec, role_matrix).flatten()
    top_indices = sims.argsort()[::-1][:top_n]

    results = []
    emp_skills_set = set(s.strip() for s in employee_skills_str.split(","))
    for idx in top_indices:
        role = INTERNAL_ROLES[idx]
        required = set(role["skills"])
        matched  = emp_skills_set & required
        gap      = required - emp_skills_set
        results.append({
            "title":      role["title"],
            "dept":       role["dept"],
            "similarity": round(float(sims[idx]), 4),
            "match_pct":  round(len(matched) / len(required) * 100, 1),
            "matched":    sorted(matched),
            "gap":        sorted(gap),
            "required":   sorted(required),
        })
    return results

# ─────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Syne, Cairo, sans-serif", color="#7a93b5", size=12),
    colorway=["#2b7fff", "#8b5cf6", "#00d4ff", "#10b981", "#f59e0b", "#ef4444", "#ec4899"],
    margin=dict(l=20, r=20, t=40, b=20),
)

# ─────────────────────────────────────────────
# AUTH UI
# ─────────────────────────────────────────────
def render_login_page():
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <div style="font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
                    background:linear-gradient(135deg,#e8f0ff 30%,#00d4ff 100%);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; line-height:1.1; margin-bottom:0.5rem;">
            🧠 AI Talent Engine
        </div>
        <div style="color:#3d5475; font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                    letter-spacing:0.15em; text-transform:uppercase;">
            Enterprise HR Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_reg = st.tabs(["🔐 تسجيل الدخول", "✨ حساب جديد"])

    with tab_login:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        email = st.text_input("البريد الإلكتروني", placeholder="you@company.com", key="login_email")
        password = st.text_input("كلمة المرور", type="password", placeholder="••••••••", key="login_pw")
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("تسجيل الدخول ←", use_container_width=True, key="btn_login"):
            if not email or not password:
                st.markdown('<div class="alert-box alert-error">⚠️ يرجى إدخال البريد الإلكتروني وكلمة المرور.</div>', unsafe_allow_html=True)
            else:
                user = login_user(email.strip(), password)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["user_id"]    = user[0]
                    st.session_state["user_email"] = user[1]
                    st.session_state["user_name"]  = user[2] or user[1]
                    st.session_state["user_role"]  = user[3]
                    st.rerun()
                else:
                    st.markdown('<div class="alert-box alert-error">❌ بيانات الدخول غير صحيحة.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top:1.5rem; padding:1rem; background:rgba(43,127,255,0.06);
                    border:1px solid rgba(43,127,255,0.15); border-radius:8px;
                    font-size:0.78rem; color:#4a6fa0; font-family:'JetBrains Mono',monospace;">
            <div style="color:#2b7fff; margin-bottom:0.4rem; font-weight:600;">🔑 Admin Demo</div>
            <div>Email: kareemeltemsah7@gmail.com</div>
            <div>Pass: temsah1</div>
        </div>
        """, unsafe_allow_html=True)

    with tab_reg:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        r_name  = st.text_input("الاسم الكامل", placeholder="اسمك الكامل", key="reg_name")
        r_email = st.text_input("البريد الإلكتروني", placeholder="you@company.com", key="reg_email")
        r_pw    = st.text_input("كلمة المرور", type="password", placeholder="8 أحرف على الأقل", key="reg_pw")
        r_pw2   = st.text_input("تأكيد كلمة المرور", type="password", placeholder="أعد الكتابة", key="reg_pw2")
        if st.button("إنشاء حساب ←", use_container_width=True, key="btn_reg"):
            if not all([r_name, r_email, r_pw, r_pw2]):
                st.markdown('<div class="alert-box alert-error">⚠️ يرجى ملء جميع الحقول.</div>', unsafe_allow_html=True)
            elif r_pw != r_pw2:
                st.markdown('<div class="alert-box alert-error">❌ كلمتا المرور غير متطابقتين.</div>', unsafe_allow_html=True)
            elif len(r_pw) < 6:
                st.markdown('<div class="alert-box alert-error">❌ كلمة المرور قصيرة جداً (6 أحرف على الأقل).</div>', unsafe_allow_html=True)
            else:
                ok, msg = register_user(r_email.strip(), r_pw, r_name.strip())
                if ok:
                    st.markdown(f'<div class="alert-box alert-success">✅ {msg} يمكنك تسجيل الدخول الآن.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-box alert-error">❌ {msg}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1.2rem 0 1rem;">
            <div class="sidebar-logo">🧠 AI Talent Engine</div>
            <div class="sidebar-sub">v2.0 · Enterprise HR Platform</div>
        </div>
        """, unsafe_allow_html=True)

        role_badge = "🛡️ Admin" if st.session_state.get("user_role") == "admin" else "👤 User"
        role_color = "#2b7fff" if st.session_state.get("user_role") == "admin" else "#8b5cf6"
        st.markdown(f"""
        <div style="background:rgba(43,127,255,0.06); border:1px solid var(--border);
                    border-radius:10px; padding:0.9rem 1rem; margin-bottom:1.2rem;">
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:0.88rem;
                        color:var(--text-primary);">{st.session_state.get('user_name','')}</div>
            <div style="font-size:0.72rem; color:var(--text-muted); margin-top:2px; font-family:'JetBrains Mono',monospace;">
                {st.session_state.get('user_email','')}
            </div>
            <div style="margin-top:0.5rem;">
                <span style="background:rgba(43,127,255,0.15); color:{role_color};
                             border:1px solid {role_color}55; border-radius:100px;
                             padding:0.15rem 0.6rem; font-size:0.7rem; font-family:'JetBrains Mono',monospace;">
                    {role_badge}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-sub" style="margin-bottom:0.5rem; padding:0 0.2rem;">NAVIGATION</div>', unsafe_allow_html=True)

        pages = [
            ("📊", "HR Executive Summary",    "hr_summary"),
            ("🔮", "Flight Risk Predictor",   "risk_predictor"),
            ("🚀", "Career Mobility Engine",  "career_mobility"),
        ]
        if st.session_state.get("user_role") == "admin":
            pages.append(("🛡️", "Admin Dashboard", "admin_panel"))

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = "hr_summary"

        for icon, label, key in pages:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state["current_page"] = key
                st.rerun()

        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🚪  تسجيل الخروج", use_container_width=True, key="btn_logout"):
            for k in ["authenticated", "user_id", "user_email", "user_name", "user_role", "current_page"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("""
        <div style="margin-top:1rem; text-align:center; font-size:0.65rem;
                    color:var(--text-muted); font-family:'JetBrains Mono',monospace;">
            © 2025 AI Talent Engine<br>Powered by ML + NLP
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 1 — HR Executive Summary
# ─────────────────────────────────────────────
def page_hr_summary(df: pd.DataFrame):
    st.markdown("""
    <div class="page-hero">
        <h1>📊 HR Executive Summary</h1>
        <p>نظرة شاملة على صحة القوى العاملة — Workforce Intelligence at a Glance</p>
    </div>
    """, unsafe_allow_html=True)

    total     = len(df)
    avg_sat   = df["Satisfaction_Score"].mean()
    at_risk   = df["Churn_Risk"].sum()
    ret_rate  = (1 - at_risk / total) * 100
    avg_eval  = df["Last_Evaluation_Score"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "👥", f"{total:,}", "إجمالي الموظفين", "+12%", True),
        (k2, "😊", f"{avg_sat:.2f}", "متوسط الرضا الوظيفي", "+0.03", True),
        (k3, "🏆", f"{ret_rate:.1f}%", "معدل الاحتفاظ", "-1.2%", False),
        (k4, "⚠️", f"{at_risk:,}", "موظفون في خطر", "+5%", False),
        (k5, "⭐", f"{avg_eval:.2f}", "متوسط تقييم الأداء", "+0.02", True),
    ]
    for col, icon, val, label, delta, positive in kpis:
        delta_class = "delta-pos" if positive else "delta-neg"
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <span class="stat-icon">{icon}</span>
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
                <div class="stat-delta {delta_class}">{delta} vs last quarter</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">🏢 توزيع الموظفين حسب القسم</div>', unsafe_allow_html=True)
        dept_counts = df["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Count"]
        fig_dept = px.bar(
            dept_counts, x="Count", y="Department", orientation="h",
            color="Count", color_continuous_scale=["#1e2d42", "#2b7fff"],
            title=""
        )
        fig_dept.update_traces(marker_line_width=0)
        fig_dept.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e2d42", linecolor="#1e2d42"),
            yaxis=dict(gridcolor="#1e2d42", linecolor="#1e2d42", tickfont=dict(size=11))
        )
        st.plotly_chart(fig_dept, use_container_width=True, key="bar_dept_counts")

    with col_b:
        st.markdown('<div class="section-title">⚠️ معدل المخاطر حسب القسم</div>', unsafe_allow_html=True)
        risk_dept = df.groupby("Department")["Churn_Risk"].agg(["sum", "count"]).reset_index()
        risk_dept.columns = ["Department", "At_Risk", "Total"]
        risk_dept["Risk_Rate"] = (risk_dept["At_Risk"] / risk_dept["Total"] * 100).round(1)
        risk_dept = risk_dept.sort_values("Risk_Rate", ascending=False)
        fig_risk = px.bar(
            risk_dept, x="Risk_Rate", y="Department", orientation="h",
            color="Risk_Rate", color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            labels={"Risk_Rate": "معدل المخاطر %"}, title=""
        )
        fig_risk.update_traces(marker_line_width=0)
        fig_risk.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e2d42", linecolor="#1e2d42"),
            yaxis=dict(gridcolor="#1e2d42", linecolor="#1e2d42", tickfont=dict(size=11))
        )
        st.plotly_chart(fig_risk, use_container_width=True, key="bar_risk_rates")

    col_c, col_d = st.columns([3, 2])

    with col_c:
        st.markdown('<div class="section-title">🔗 مصفوفة الارتباط (Correlation Heatmap)</div>', unsafe_allow_html=True)
        numeric_cols = ["Tenure_Months", "Satisfaction_Score", "Avg_Monthly_Hours",
                        "Last_Evaluation_Score", "Salary_Tier_Num", "Churn_Risk"]
        corr = df[numeric_cols].corr().round(3)
        nice_labels = ["مدة الخدمة", "الرضا الوظيفي", "ساعات العمل",
                       "تقييم الأداء", "مستوى الراتب", "خطر المغادرة"]
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=nice_labels, y=nice_labels,
            colorscale=[[0,"#ef4444"],[0.5,"#1e2d42"],[1,"#2b7fff"]],
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=11, family="JetBrains Mono"),
            zmin=-1, zmax=1,
        ))
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_heat, use_container_width=True, key="heat_correlation")

    with col_d:
        st.markdown('<div class="section-title">💰 توزيع مستويات الرواتب</div>', unsafe_allow_html=True)
        sal_counts = df["Salary_Tier"].value_counts().reset_index()
        sal_counts.columns = ["Tier", "Count"]
        fig_sal = px.pie(
            sal_counts, names="Tier", values="Count",
            color_discrete_sequence=["#2b7fff", "#8b5cf6", "#00d4ff", "#10b981", "#f59e0b"],
            hole=0.55,
        )
        fig_sal.update_traces(textfont=dict(family="JetBrains Mono"), textinfo="percent+label")
        fig_sal.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
        st.plotly_chart(fig_sal, use_container_width=True, key="pie_salary_tiers")

    st.markdown('<div class="section-title">📈 الرضا الوظيفي مقابل ساعات العمل (عينة 500 موظف)</div>', unsafe_allow_html=True)
    sample = df.sample(500, random_state=1)
    fig_scatter = px.scatter(
        sample,
        x="Avg_Monthly_Hours", y="Satisfaction_Score",
        color="Churn_Risk",
        color_discrete_map={0: "#10b981", 1: "#ef4444"},
        size="Last_Evaluation_Score",
        hover_data=["Department", "Salary_Tier", "Tenure_Months"],
        labels={"Avg_Monthly_Hours": "ساعات العمل الشهرية",
                "Satisfaction_Score": "درجة الرضا",
                "Churn_Risk": "خطر المغادرة"},
        opacity=0.7,
    )
    fig_scatter.update_traces(marker=dict(line=dict(width=0)))
    fig_scatter.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11))
    )
    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_sat_hours")

# ─────────────────────────────────────────────
# PAGE 2 — Flight Risk Predictor
# ─────────────────────────────────────────────
def page_risk_predictor(df: pd.DataFrame):
    st.markdown("""
    <div class="page-hero">
        <h1>🔮 Flight Risk Predictor</h1>
        <p>أدخل بيانات الموظف واحصل على تقييم فوري لاحتمالية مغادرته مع تحليل الأسباب</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("⚙️ جارٍ تحميل نموذج الذكاء الاصطناعي..."):
        model, le_dept, feature_names, acc = train_churn_model(df)

    st.markdown(f"""
    <div class="alert-box alert-info">
        🤖 نموذج <b>Random Forest (300 شجرة)</b> تم تدريبه على {len(df):,} موظف —
        دقة النموذج: <b>{acc*100:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="engine-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 بيانات الموظف</div>', unsafe_allow_html=True)

        dept_choice = st.selectbox("القسم", options=DEPARTMENTS, key="rp_dept")
        tenure_val  = st.slider("مدة الخدمة (أشهر)", 1, 144, 24, key="rp_tenure")
        sat_val     = st.slider("درجة الرضا الوظيفي", 0.1, 1.0, 0.65, 0.01, key="rp_sat")
        hours_val   = st.slider("متوسط ساعات العمل الشهرية", 120, 310, 170, key="rp_hours")
        eval_val    = st.slider("درجة تقييم الأداء", 0.2, 1.0, 0.70, 0.01, key="rp_eval")
        sal_choice  = st.selectbox("مستوى الراتب", SALARY_TIERS, index=1, key="rp_sal")

        predict_btn = st.button("🔮 تحليل احتمالية المغادرة", use_container_width=True, key="btn_predict")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if predict_btn:
            sal_num = SALARY_TIERS.index(sal_choice) + 1
            try:
                dept_enc = le_dept.transform([dept_choice])[0]
            except ValueError:
                dept_enc = 0

            X_input = pd.DataFrame([[
                tenure_val, sat_val, hours_val, eval_val, sal_num, dept_enc
            ]], columns=feature_names)

            prob = model.predict_proba(X_input)[0][1]
            risk_pct = round(prob * 100, 1)

            if risk_pct >= 70:
                risk_level = "مرتفع جداً"
                risk_class = "risk-high"
                gauge_color = "#ef4444"
            elif risk_pct >= 45:
                risk_level = "متوسط"
                risk_class = "risk-medium"
                gauge_color = "#f59e0b"
            else:
                risk_level = "منخفض"
                risk_class = "risk-low"
                gauge_color = "#10b981"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                number={"suffix": "%", "font": {"family": "JetBrains Mono", "size": 42, "color": gauge_color}},
                delta={"reference": 30, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#3d5475",
                             "tickfont": {"family": "JetBrains Mono", "size": 10}},
                    "bar": {"color": gauge_color, "thickness": 0.22},
                    "bgcolor": "#0e1520",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,   40],  "color": "rgba(16,185,129,0.10)"},
                        {"range": [40,  70],  "color": "rgba(245,158,11,0.10)"},
                        {"range": [70,  100], "color": "rgba(239,68,68,0.10)"},
                    ],
                    "threshold": {"line": {"color": gauge_color, "width": 3}, "value": risk_pct},
                },
                title={"text": f"احتمالية المغادرة", "font": {"family": "Syne", "size": 14, "color": "#7a93b5"}},
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
                                    margin=dict(l=20, r=20, t=30, b=10),
                                    font=dict(family="Syne, Cairo"))
            st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_risk")

            st.markdown(f"""
            <div class="engine-card" style="text-align:center; margin-top:-0.5rem;">
                <div style="font-family:'Syne',sans-serif; font-size:0.85rem; color:var(--text-secondary);">مستوى الخطر</div>
                <div class="{risk_class}" style="font-size:1.6rem; margin: 0.4rem 0;">{risk_level}</div>
                <div style="font-size:0.8rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">
                    Confidence: {max(prob, 1-prob)*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-title" style="margin-top:1rem;">🔍 أهم العوامل المؤثرة</div>', unsafe_allow_html=True)
            nice_feat_names = {
                "Tenure_Months": "مدة الخدمة",
                "Satisfaction_Score": "الرضا الوظيفي",
                "Avg_Monthly_Hours": "ساعات العمل",
                "Last_Evaluation_Score": "تقييم الأداء",
                "Salary_Tier_Num": "مستوى الراتب",
                "Dept_Encoded": "القسم",
            }
            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": [nice_feat_names.get(f, f) for f in feature_names],
                "Importance": importances,
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                feat_df, x="Importance", y="Feature", orientation="h",
                color="Importance",
                color_continuous_scale=["#1e2d42", "#2b7fff", "#00d4ff"],
            )
            fig_imp.update_traces(marker_line_width=0)
            fig_imp.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True, key="bar_feature_importance")

        else:
            st.markdown("""
            <div class="engine-card" style="text-align:center; padding:3rem 2rem; opacity:0.6;">
                <div style="font-size:3rem; margin-bottom:1rem;">🔮</div>
                <div style="font-family:'Syne',sans-serif; font-size:1rem; color:var(--text-secondary);">
                    أدخل بيانات الموظف واضغط على "تحليل"<br>للحصول على التقييم الفوري
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📋 موظفو الخطر المرتفع (Top 20 أعلى خطراً)</div>', unsafe_allow_html=True)
    top_risk = df.nlargest(20, "Churn_Score_Raw")[
        ["Employee_ID", "Department", "Tenure_Months", "Satisfaction_Score",
         "Avg_Monthly_Hours", "Salary_Tier", "Churn_Score_Raw"]
    ].copy()
    top_risk["مستوى الخطر"] = top_risk["Churn_Score_Raw"].apply(
        lambda x: "🔴 مرتفع" if x > 0.65 else ("🟡 متوسط" if x > 0.45 else "🟢 منخفض")
    )
    top_risk.columns = ["الرقم الوظيفي", "القسم", "الخدمة (أشهر)", "الرضا",
                         "ساعات العمل", "مستوى الراتب", "درجة الخطر", "التصنيف"]
    st.dataframe(
        top_risk.style.background_gradient(subset=["درجة الخطر"], cmap="RdYlGn_r"),
        use_container_width=True, height=380
    )

# ─────────────────────────────────────────────
# PAGE 3 — Career Mobility Engine
# ─────────────────────────────────────────────
def page_career_mobility(df: pd.DataFrame):
    st.markdown("""
    <div class="page-hero">
        <h1>🚀 Smart Career Mobility Engine</h1>
        <p>محرك الانتقال الوظيفي الذكي — اكتشف أفضل المسارات الداخلية بناءً على مهاراتك</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🧠 تحليل مصفوفة المهارات..."):
        build_career_engine()

    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="engine-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 ملف الموظف</div>', unsafe_allow_html=True)

        mode = st.radio("اختر طريقة الإدخال", ["اختر موظفاً موجوداً", "أدخل مهاراتك يدوياً"],
                        horizontal=True, key="cm_mode")

        if mode == "اختر موظفاً موجوداً":
            sel_dept = st.selectbox("القسم", DEPARTMENTS, key="cm_dept")
            dept_df = df[df["Department"] == sel_dept].sample(min(20, len(df[df["Department"] == sel_dept])), random_state=7)
            sel_emp = st.selectbox("اختر الموظف", dept_df["Employee_ID"].tolist(), key="cm_emp")
            emp_row = df[df["Employee_ID"] == sel_emp].iloc[0]
            skills_input = emp_row["Skills"]
            st.markdown(f"""
            <div style="margin-top:0.8rem;">
                <div style="font-size:0.8rem; color:var(--text-secondary); margin-bottom:0.5rem; font-family:'Syne',sans-serif; font-weight:600;">المهارات الحالية</div>
                {"".join(f'<span class="skill-tag">{s.strip()}</span>' for s in skills_input.split(","))}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-top:0.8rem; display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
                <div style="background:var(--bg-base); border-radius:6px; padding:0.6rem 0.8rem;">
                    <div style="font-size:0.7rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">SATISFACTION</div>
                    <div style="font-size:1rem; color:var(--accent-cyan); font-weight:700; font-family:'JetBrains Mono',monospace;">{emp_row['Satisfaction_Score']:.2f}</div>
                </div>
                <div style="background:var(--bg-base); border-radius:6px; padding:0.6rem 0.8rem;">
                    <div style="font-size:0.7rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">TENURE</div>
                    <div style="font-size:1rem; color:var(--accent-cyan); font-weight:700; font-family:'JetBrains Mono',monospace;">{emp_row['Tenure_Months']} mos</div>
                </div>
                <div style="background:var(--bg-base); border-radius:6px; padding:0.6rem 0.8rem;">
                    <div style="font-size:0.7rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">EVALUATION</div>
                    <div style="font-size:1rem; color:var(--accent-green); font-weight:700; font-family:'JetBrains Mono',monospace;">{emp_row['Last_Evaluation_Score']:.2f}</div>
                </div>
                <div style="background:var(--bg-base); border-radius:6px; padding:0.6rem 0.8rem;">
                    <div style="font-size:0.7rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">CHURN RISK</div>
                    <div style="font-size:1rem; font-weight:700; font-family:'JetBrains Mono',monospace;
                                color:{'#ef4444' if emp_row['Churn_Risk']==1 else '#10b981'};">
                        {'HIGH ⚠️' if emp_row['Churn_Risk']==1 else 'LOW ✓'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pool_all = sorted(set(s for pool in SKILLS_POOL.values() for s in pool))
            chosen_skills = st.multiselect("اختر مهاراتك", pool_all, default=["Python", "SQL", "Machine Learning"], key="cm_custom_skills")
            skills_input = ", ".join(chosen_skills) if chosen_skills else ""

        analyze_btn = st.button("🚀 تحليل المسار الوظيفي", use_container_width=True, key="btn_career")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if analyze_btn and skills_input:
            results = recommend_roles(skills_input, top_n=3)

            st.markdown('<div class="section-title">🎯 أفضل 3 وظائف داخلية مقترحة</div>', unsafe_allow_html=True)

            medals = ["🥇", "🥈", "🥉"]
            colors = ["#2b7fff", "#8b5cf6", "#10b981"]

            for i, role in enumerate(results):
                match_color = colors[i % len(colors)]
                sim_pct = round(role["similarity"] * 100, 1)

                with st.container():
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:start;">
                            <div>
                                <span style="font-size:1.1rem; margin-right:0.4rem;">{medals[i]}</span>
                                <h4 style="display:inline; color:{match_color} !important;">{role['title']}</h4>
                                <div style="margin-top:0.3rem;">
                                    <span class="badge badge-blue">{role['dept']}</span>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem;
                                            font-weight:700; color:{match_color};">{role['match_pct']}%</div>
                                <div style="font-size:0.68rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">تطابق المهارات</div>
                            </div>
                        </div>
                        <div class="progress-bar-wrap" style="margin:0.8rem 0 0.6rem;">
                            <div class="progress-bar-fill" style="width:{role['match_pct']}%;
                                 background:linear-gradient(90deg, {match_color}99, {match_color});"></div>
                        </div>
                        <div style="font-size:0.75rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace;
                                    margin-bottom:0.6rem;">
                            Cosine Similarity: {sim_pct}%
                        </div>
                    """, unsafe_allow_html=True)

                    if role["matched"]:
                        st.markdown(f"""
                        <div style="margin-bottom:0.5rem;">
                            <div style="font-size:0.72rem; color:var(--accent-green); font-family:'Syne',sans-serif;
                                        font-weight:600; margin-bottom:0.3rem;">✅ مهاراتك المطابقة</div>
                            {"".join(f'<span class="skill-tag">{s}</span>' for s in role['matched'])}
                        </div>
                        """, unsafe_allow_html=True)

                    if role["gap"]:
                        st.markdown(f"""
                        <div>
                            <div style="font-size:0.72rem; color:var(--accent-orange); font-family:'Syne',sans-serif;
                                        font-weight:600; margin-bottom:0.3rem;">📚 فجوة المهارات المطلوبة</div>
                            {"".join(f'<span class="skill-gap-tag">{s}</span>' for s in role['gap'])}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-title" style="margin-top:1rem;">📡 تحليل تغطية المهارات</div>', unsafe_allow_html=True)
            categories = ["Match %", "Cosine Sim %", "Gap Count (inv)", "Required Skills", "Score"]
            fig_radar = go.Figure()
            for i, role in enumerate(results):
                gap_inv = max(0, 100 - len(role["gap"]) * 15)
                req_score = min(100, len(role["required"]) * 10)
                vals = [
                    role["match_pct"],
                    role["similarity"] * 100,
                    gap_inv,
                    req_score,
                    (role["match_pct"] + role["similarity"] * 100) / 2,
                ]
                vals += [vals[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=role["title"],
                    line=dict(color=colors[i % len(colors)], width=2),
                    fillcolor=f"{colors[i % len(colors)]}22",
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1e2d42",
                                   tickfont=dict(family="JetBrains Mono", size=9)),
                    angularaxis=dict(gridcolor="#1e2d42"),
                ),
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
                height=320,
                showlegend=True,
                legend=dict(font=dict(size=10, family="Syne"), orientation="h",
                            yanchor="bottom", y=1.05),
            )
            st.plotly_chart(fig_radar, use_container_width=True, key="radar_skills")

        elif analyze_btn and not skills_input:
            st.markdown('<div class="alert-box alert-error">⚠️ يرجى إدخال مهارات أو اختيار موظف أولاً.</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="engine-card" style="text-align:center; padding:4rem 2rem; opacity:0.55; margin-top:2rem;">
                <div style="font-size:3.5rem; margin-bottom:1rem;">🚀</div>
                <div style="font-family:'Syne',sans-serif; font-size:1rem; color:var(--text-secondary);">
                    اختر موظفاً أو أدخل مهاراتك<br>واضغط "تحليل المسار الوظيفي"
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 4 — Admin Dashboard (مع إدارة المستخدمين الكاملة)
# ─────────────────────────────────────────────
def page_admin(df: pd.DataFrame):
    st.markdown("""
    <div class="page-hero" style="border-color:#2b7fff44;">
        <h1>🛡️ Admin Control Center</h1>
        <p>إحصائيات النظام الشاملة وإدارة المستخدمين — System-wide Analytics & User Management</p>
    </div>
    """, unsafe_allow_html=True)

    users_df = get_all_users()
    total_users = len(users_df)
    admin_count = (users_df["role"] == "admin").sum()
    regular_count = total_users - admin_count

    s1, s2, s3, s4 = st.columns(4)
    for col, icon, val, label in [
        (s1, "👥", str(total_users), "إجمالي المستخدمين"),
        (s2, "🛡️", str(admin_count), "المسؤولون"),
        (s3, "👤", str(regular_count), "المستخدمون العاديون"),
        (s4, "💾", f"{len(df):,}", "سجلات الموظفين"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <span class="stat-icon">{icon}</span>
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">👥 إدارة المستخدمين (حذف / ترقية / خفض)</div>', unsafe_allow_html=True)

    display_users = users_df.copy()
    display_users.columns = ["ID", "البريد الإلكتروني", "الاسم الكامل", "الدور", "تاريخ الإنشاء", "آخر دخول"]

    st.markdown("#### 🔧 عمليات سريعة على مستخدم معين")

    col_sel, col_op = st.columns([2,1])
    with col_sel:
        user_to_manage = st.selectbox(
            "اختر المستخدم",
            options=users_df["id"].tolist(),
            format_func=lambda x: f"{users_df[users_df['id']==x]['email'].iloc[0]} ({users_df[users_df['id']==x]['role'].iloc[0]})"
        )
    selected_user_role = users_df[users_df["id"] == user_to_manage]["role"].iloc[0]
    selected_user_email = users_df[users_df["id"] == user_to_manage]["email"].iloc[0]

    with col_op:
        if st.button("🗑️ حذف هذا المستخدم", key="delete_selected"):
            ok, msg = delete_user(user_to_manage, st.session_state["user_id"])
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        if selected_user_role != "admin":
            if st.button("⭐ ترقية إلى Admin", key="promote_selected"):
                ok, msg = promote_to_admin(user_to_manage)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            if st.button("⬇️ خفض إلى User", key="demote_selected"):
                ok, msg = demote_to_user(user_to_manage, st.session_state["user_id"], st.session_state["user_role"])
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("---")
    st.markdown("#### 📋 قائمة جميع المستخدمين")
    st.dataframe(display_users, use_container_width=True, height=300)

    st.markdown("---")
    st.markdown('<div class="section-title">📊 إحصائيات قاعدة البيانات</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        dept_sal = df.groupby(["Department", "Salary_Tier"]).size().reset_index(name="Count")
        pivot = dept_sal.pivot(index="Department", columns="Salary_Tier", values="Count").fillna(0)
        fig_h2 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0,"#0e1520"],[1,"#2b7fff"]],
            texttemplate="%{z:.0f}",
            textfont=dict(size=10, family="JetBrains Mono"),
        ))
        fig_h2.update_layout(**PLOTLY_LAYOUT, title="توزيع الأقسام × مستوى الراتب", height=420)
        st.plotly_chart(fig_h2, use_container_width=True, key="heatmap_dept_salary")

    with col_b:
        model, le_dept, feature_names, acc = train_churn_model(df)
        importances = model.feature_importances_
        nice_names = ["مدة الخدمة", "الرضا", "ساعات العمل", "تقييم الأداء", "مستوى الراتب", "القسم"]
        fig_imp_admin = px.pie(
            names=nice_names,
            values=importances,
            color_discrete_sequence=["#2b7fff","#8b5cf6","#00d4ff","#10b981","#f59e0b","#ef4444"],
            hole=0.4,
            title=f"توزيع أهمية المتغيرات — دقة النموذج: {acc*100:.1f}%",
        )
        fig_imp_admin.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig_imp_admin, use_container_width=True, key="pie_feature_importance")

    st.markdown('<div class="section-title">🗄️ معاينة قاعدة بيانات الموظفين</div>', unsafe_allow_html=True)
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        f_dept = st.selectbox("تصفية: القسم", ["الكل"] + DEPARTMENTS, key="adm_fdept")
    with col_filter2:
        f_risk = st.selectbox("تصفية: الخطر", ["الكل", "مرتفع", "منخفض"], key="adm_frisk")
    with col_filter3:
        f_sal = st.selectbox("تصفية: الراتب", ["الكل"] + SALARY_TIERS, key="adm_fsal")

    filtered = df.copy()
    if f_dept != "الكل":     filtered = filtered[filtered["Department"] == f_dept]
    if f_risk == "مرتفع":   filtered = filtered[filtered["Churn_Risk"] == 1]
    if f_risk == "منخفض":   filtered = filtered[filtered["Churn_Risk"] == 0]
    if f_sal != "الكل":      filtered = filtered[filtered["Salary_Tier"] == f_sal]

    display_cols = ["Employee_ID", "Department", "Tenure_Months", "Satisfaction_Score",
                    "Avg_Monthly_Hours", "Last_Evaluation_Score", "Salary_Tier", "Churn_Risk"]
    st.dataframe(filtered[display_cols].head(100), use_container_width=True, height=380)
    st.markdown(f'<div style="color:var(--text-muted); font-size:0.78rem; font-family:JetBrains Mono,monospace; margin-top:0.5rem;">عرض أول 100 من {len(filtered):,} سجل</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN APPLICATION CONTROLLER
# ─────────────────────────────────────────────
def main():
    init_db()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        _, center, _ = st.columns([1, 2, 1])
        with center:
            render_login_page()
        return

    with st.spinner("🔄 جارٍ تحميل قاعدة البيانات..."):
        df = generate_employee_data(2000)

    render_sidebar()

    page = st.session_state.get("current_page", "hr_summary")

    try:
        if page == "hr_summary":
            page_hr_summary(df)
        elif page == "risk_predictor":
            page_risk_predictor(df)
        elif page == "career_mobility":
            page_career_mobility(df)
        elif page == "admin_panel":
            if st.session_state.get("user_role") == "admin":
                page_admin(df)
            else:
                st.error("⛔ غير مصرح. هذه الصفحة للمسؤولين فقط.")
        else:
            page_hr_summary(df)
    except Exception as e:
        st.error(f"❌ حدث خطأ: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()