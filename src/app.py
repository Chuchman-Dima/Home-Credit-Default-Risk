"""
app.py — Streamlit UI для Home Credit Default Risk

Запуск: streamlit run streamlit_app/app.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Home Credit Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  .main { background-color: #0f1117; }

  .metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 1px solid #3a3a5e;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-card .label {
    color: #8888aa;
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }
  .metric-card .value {
    color: #ffffff;
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
  }
  .metric-card .sub {
    color: #6666aa;
    font-size: 11px;
    margin-top: 4px;
  }

  .risk-low    { color: #22c55e !important; }
  .risk-medium { color: #f59e0b !important; }
  .risk-high   { color: #ef4444 !important; }

  .score-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
  }
  .badge-low    { background: #14532d; color: #22c55e; }
  .badge-medium { background: #451a03; color: #f59e0b; }
  .badge-high   { background: #450a0a; color: #ef4444; }

  .section-header {
    color: #a0a0cc;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 20px 0 8px;
    border-bottom: 1px solid #2a2a4a;
    padding-bottom: 6px;
  }

  div[data-testid="stSidebar"] {
    background-color: #12121e;
    border-right: 1px solid #2a2a4a;
  }
  div[data-testid="stSidebar"] .stMarkdown h1,
  div[data-testid="stSidebar"] .stMarkdown h2,
  div[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "credit_scoring_lgbm.pkl"
    )
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_metadata():
    meta_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "model_metadata.json"
    )
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_feature_importance():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "feature_importance.csv"
    )
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


pipeline = load_model()
metadata = load_metadata()
importance_df = load_feature_importance()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def risk_label(prob: float):
    if prob < 0.3:
        return "НИЗЬКИЙ", "badge-low", "🟢"
    elif prob < 0.6:
        return "СЕРЕДНІЙ", "badge-medium", "🟡"
    else:
        return "ВИСОКИЙ", "badge-high", "🔴"


def get_credit_score(prob: float) -> int:
    """Convert probability to 300-850 credit score."""
    return int(300 + (1 - prob) * 550)


def make_gauge(prob: float):
    """Plotly gauge chart for default probability."""
    score = get_credit_score(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Кредитний скор", "font": {"size": 18, "color": "#aaaacc"}},
        delta={"reference": 580, "decreasing": {"color": "#ef4444"},
               "increasing": {"color": "#22c55e"}},
        gauge={
            "axis": {"range": [300, 850], "tickwidth": 1,
                     "tickcolor": "#666", "tickfont": {"color": "#999"}},
            "bar": {"color": "#6366f1", "thickness": 0.3},
            "bgcolor": "#1a1a2e",
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps": [
                {"range": [300, 500], "color": "#450a0a"},
                {"range": [500, 620], "color": "#451a03"},
                {"range": [620, 720], "color": "#1a2e1a"},
                {"range": [720, 850], "color": "#0a2e0a"},
            ],
            "threshold": {
                "line": {"color": "#fff", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
        number={"font": {"size": 40, "color": "#ffffff"}},
    ))
    fig.update_layout(
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        height=280,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def make_prob_bar(prob: float):
    color = "#22c55e" if prob < 0.3 else "#f59e0b" if prob < 0.6 else "#ef4444"
    fig = go.Figure(go.Bar(
        x=[prob * 100], y=["Ймовірність"],
        orientation="h",
        marker_color=color,
        text=[f"{prob*100:.1f}%"],
        textposition="outside",
        textfont={"color": "#fff", "size": 16},
    ))
    fig.add_vline(x=30, line_dash="dash", line_color="rgba(34,197,94,0.5)")
    fig.add_vline(x=60, line_dash="dash", line_color="rgba(239,68,68,0.5)")
    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, tickfont={"color": "#888"}),
        yaxis=dict(showgrid=False, tickfont={"color": "#888"}),
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        height=100,
        margin=dict(t=10, b=10, l=10, r=60),
        font={"color": "#ccc"},
    )
    return fig


def predict_single(data: dict):
    """Build a single-row DataFrame with all required features and predict."""
    # Age
    age = data.get("age", 35)
    days_birth = -age * 365
    days_employed = -data.get("years_employed", 5) * 365 if data.get("years_employed", 5) > 0 else 365243

    row = {
        "sk_id_curr": 0,
        "target": 0,
        "name_contract_type": data.get("contract_type", "Cash loans"),
        "code_gender": data.get("gender", "M"),
        "flag_own_car": "Y" if data.get("own_car") else "N",
        "flag_own_realty": "Y" if data.get("own_realty") else "N",
        "cnt_children": data.get("children", 0),
        "amt_income_total": data.get("income", 100000),
        "amt_credit": data.get("credit", 200000),
        "amt_annuity": data.get("annuity", 15000),
        "amt_goods_price": data.get("goods_price", 180000),
        "name_income_type": data.get("income_type", "Working"),
        "name_education_type": data.get("education", "Secondary / secondary special"),
        "name_family_status": data.get("family_status", "Married"),
        "name_housing_type": data.get("housing", "House / apartment"),
        "region_population_relative": data.get("region_pop", 0.02),
        "days_birth": days_birth,
        "days_employed": days_employed,
        "days_registration": -data.get("years_registered", 5) * 365,
        "days_id_publish": -data.get("years_id", 5) * 365,
        "flag_mobil": 1,
        "flag_emp_phone": 1,
        "flag_work_phone": 0,
        "flag_cont_mobile": 1,
        "flag_phone": 0,
        "flag_email": data.get("flag_email", 0),
        "occupation_type": data.get("occupation", "Laborers"),
        "cnt_fam_members": data.get("children", 0) + 2,
        "region_rating_client": 2,
        "region_rating_client_w_city": 2,
        "weekday_appr_process_start": "WEDNESDAY",
        "hour_appr_process_start": 10,
        "reg_region_not_live_region": 0,
        "reg_region_not_work_region": 0,
        "live_region_not_work_region": 0,
        "reg_city_not_live_city": 0,
        "reg_city_not_work_city": 0,
        "live_city_not_work_city": 0,
        "organization_type": data.get("org_type", "Business Entity Type 3"),
        "ext_source_1": data.get("ext_source_1", 0.5),
        "ext_source_2": data.get("ext_source_2", 0.5),
        "ext_source_3": data.get("ext_source_3", 0.5),
        "apartments_avg": None, "basementarea_avg": None,
        "years_beginexpluatation_avg": None, "years_build_avg": None,
        "commonarea_avg": None, "elevators_avg": None,
        "entrances_avg": None, "floorsmax_avg": None,
        "floorsmin_avg": None, "landarea_avg": None,
        "livingapartments_avg": None, "livingarea_avg": None,
        "nonlivingapartments_avg": None, "nonlivingarea_avg": None,
        "totalarea_mode": None, "emergencystate_mode": None,
        "obs_30_cnt_social_circle": 0, "def_30_cnt_social_circle": 0,
        "obs_60_cnt_social_circle": 0, "def_60_cnt_social_circle": 0,
        "days_last_phone_change": -300,
        "flag_document_2": 0, "flag_document_3": 1,
        "flag_document_4": 0, "flag_document_5": 0,
        "flag_document_6": 0, "flag_document_7": 0,
        "flag_document_8": 0, "flag_document_9": 0,
        "flag_document_11": 0, "flag_document_18": 0,
        "flag_document_19": 0, "flag_document_20": 0,
        "flag_document_21": 0,
        # Відсутні ознаки будинку (_mode, _medi та інші категоріальні)
        "entrances_mode": None, "landarea_mode": None, "fondkapremont_mode": None,
        "floorsmin_mode": None, "nonlivingarea_medi": None, "floorsmax_mode": None,
        "nonlivingarea_mode": None, "basementarea_mode": None, "apartments_mode": None,
        "basementarea_medi": None, "floorsmax_medi": None, "livingarea_medi": None,
        "commonarea_mode": None, "elevators_medi": None, "apartments_medi": None,
        "years_beginexpluatation_medi": None, "livingapartments_medi": None,
        "years_beginexpluatation_mode": None, "entrances_medi": None, "livingarea_mode": None,
        "livingapartments_mode": None, "nonlivingapartments_medi": None, "wallsmaterial_mode": None,
        "nonlivingapartments_mode": None, "floorsmin_medi": None, "landarea_medi": None,
        "years_build_medi": None, "elevators_mode": None, "years_build_mode": None,
        "housetype_mode": None, "commonarea_medi": None,

        # Відсутні прапорці документів
        "flag_document_10": 0, "flag_document_12": 0, "flag_document_13": 0,
        "flag_document_14": 0, "flag_document_15": 0, "flag_document_16": 0,
        "flag_document_17": 0,

        # Інші відсутні ознаки
        "name_type_suite": None,
        "own_car_age": None,
        "amt_req_credit_bureau_hour": 0, "amt_req_credit_bureau_day": 0,
        "amt_req_credit_bureau_week": 0, "amt_req_credit_bureau_mon": 1,
        "amt_req_credit_bureau_qrt": 2, "amt_req_credit_bureau_year": 5,
        # feature-engineered (from bureau etc) — use neutral defaults
        "bureau_loan_count": data.get("bureau_loans", 3),
        "bureau_active_loans": data.get("bureau_active", 1),
        "bureau_closed_loans": data.get("bureau_closed", 2),
        "bureau_avg_days_credit": -500,
        "bureau_avg_days_enddate": 100,
        "bureau_total_debt": data.get("bureau_debt", 50000),
        "bureau_total_credit": data.get("bureau_credit", 100000),
        "bureau_avg_overdue": 0,
        "bureau_max_overdue": 0,
        "bureau_sum_overdue": 0,
        "bureau_debt_ratio": data.get("bureau_debt", 50000) / (data.get("bureau_credit", 100000) + 1),
        "bb_months_count": 12,
        "bb_avg_months_balance": -6,
        "bb_dpd_count": 0,
        "bb_overdue_count": 0,
        "prev_app_count": data.get("prev_apps", 2),
        "prev_app_approved": data.get("prev_approved", 1),
        "prev_app_refused": data.get("prev_refused", 0),
        "prev_avg_credit": 150000,
        "prev_avg_annuity": 12000,
        "prev_avg_application": 150000,
        "prev_avg_down_payment": 0,
        "prev_max_credit": 200000,
        "prev_avg_days_decision": -200,
        "prev_avg_days_last_due": -100,
        "prev_approval_rate": data.get("prev_approved", 1) / (data.get("prev_apps", 2) + 1),
        "pos_months_count": 12,
        "pos_avg_instalment": 24,
        "pos_avg_future": 12,
        "pos_completed": 1,
        "pos_active": 1,
        "pos_max_dpd": 0,
        "pos_avg_dpd": 0,
        "pos_max_dpd_def": 0,
        "inst_count": 24,
        "inst_avg_amount": 10000,
        "inst_total_payment": 240000,
        "inst_avg_payment_diff": 0,
        "inst_max_payment_diff": 0,
        "inst_avg_days_late": data.get("avg_days_late", 0),
        "inst_max_days_late": data.get("max_days_late", 0),
        "inst_late_count": data.get("late_count", 0),
        "inst_late_rate": data.get("late_count", 0) / 25,
        "cc_months_count": 12,
        "cc_avg_balance": data.get("cc_balance", 5000),
        "cc_max_balance": data.get("cc_balance", 5000) * 2,
        "cc_avg_credit_limit": data.get("cc_limit", 20000),
        "cc_avg_drawings": 1000,
        "cc_avg_payment": 2000,
        "cc_avg_receivable": 4000,
        "cc_max_dpd": 0,
        "cc_avg_dpd": 0,
        "cc_utilization": data.get("cc_balance", 5000) / (data.get("cc_limit", 20000) + 1),
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────
#  Sidebar navigation
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏦 Credit Risk")
    st.markdown("### Home Credit Default")
    st.markdown("---")

    page = st.radio(
        "Навігація",
        ["🔮 Скоринг клієнта", "📊 Аналіз портфеля", "📈 Метрики моделі", "ℹ️ Про модель"],
        label_visibility="hidden",
    )

    st.markdown("---")
    if metadata:
        st.markdown("**Модель:**")
        st.markdown(f"LightGBM Classifier")
        cv_auc = metadata.get("cv_auc_mean", 0)
        st.markdown(f"CV AUC: **{cv_auc:.4f}**")
        n_feat = metadata.get("n_features", 0)
        st.markdown(f"Ознак: **{n_feat}**")
    else:
        st.warning("Модель не завантажена")

    st.markdown("---")
    st.caption("Home Credit Default Risk")
    st.caption("LightGBM · SHAP · Streamlit")


# ═══════════════════════════════════════════════
# PAGE 1: Scoring
# ═══════════════════════════════════════════════
if page == "🔮 Скоринг клієнта":
    st.markdown("## 🔮 Скоринг клієнта")
    st.markdown("Заповніть параметри клієнта для оцінки кредитного ризику")

    if pipeline is None:
        st.error("❌ Модель не знайдена. Спочатку запустіть `notebooks/model.ipynb`")
        st.stop()

    # ── Input form ──
    with st.form("scoring_form"):
        # Personal
        st.markdown('<div class="section-header">👤 Персональні дані</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        age     = c1.number_input("Вік", 18, 75, 35)
        gender  = c2.selectbox("Стать", ["M", "F"])
        children= c3.number_input("Дітей", 0, 10, 0)
        family  = c4.selectbox("Сімейний стан", [
            "Married", "Single / not married", "Civil marriage", "Separated", "Widow"])

        c1, c2 = st.columns(2)
        own_car    = c1.checkbox("Має автомобіль")
        own_realty = c2.checkbox("Має нерухомість", value=True)

        # Employment
        st.markdown('<div class="section-header">💼 Зайнятість</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        income_type   = c1.selectbox("Тип доходу", [
            "Working", "Commercial associate", "Pensioner",
            "State servant", "Student", "Businessman", "Maternity leave"])
        education     = c2.selectbox("Освіта", [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"])
        occupation    = c3.selectbox("Професія", [
            "Laborers", "Core staff", "Managers", "Drivers",
            "High skill tech staff", "Accountants", "Medicine staff",
            "Security staff", "Cooking staff", "Cleaning staff",
            "Private service staff", "Low-skill Laborers", "HR staff",
            "Realty agents", "Secretaries", "IT staff", "Waiters/barmen staff"])
        years_employed = c4.number_input("Стаж роботи (рр)", 0, 50, 5)

        # Financial
        st.markdown('<div class="section-header">💰 Фінансові показники</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        income    = c1.number_input("Дохід (грн)", 10000, 5000000, 100000, step=10000)
        credit    = c2.number_input("Сума кредиту (грн)", 10000, 5000000, 200000, step=10000)
        annuity   = c3.number_input("Ануїтет/місяць", 1000, 200000, 15000, step=1000)
        goods     = c4.number_input("Вартість товару (грн)", 0, 5000000, 180000, step=10000)

        # External scores
        st.markdown('<div class="section-header">🏛️ Зовнішній скоринг</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        ext1 = c1.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01,
                          help="Зовнішній скор 1 (вищий = краще)")
        ext2 = c2.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5, 0.01,
                          help="Зовнішній скор 2")
        ext3 = c3.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01,
                          help="Зовнішній скор 3")

        # Payment history
        st.markdown('<div class="section-header">📋 Кредитна історія</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        prev_apps    = c1.number_input("Попередніх заявок", 0, 50, 2)
        prev_approved= c2.number_input("Схвалено", 0, 50, 1)
        avg_late     = c3.number_input("Середн. прострочення (дн)", 0, 180, 0)
        late_count   = c4.number_input("К-сть прострочених платежів", 0, 100, 0)

        submitted = st.form_submit_button(
            "🎯 Розрахувати скор", use_container_width=True, type="primary"
        )

    if submitted:
        input_data = {
            "age": age, "gender": gender, "children": children,
            "family_status": family, "own_car": own_car, "own_realty": own_realty,
            "income_type": income_type, "education": education,
            "occupation": occupation, "years_employed": years_employed,
            "income": income, "credit": credit, "annuity": annuity,
            "goods_price": goods, "ext_source_1": ext1, "ext_source_2": ext2,
            "ext_source_3": ext3, "prev_apps": prev_apps,
            "prev_approved": prev_approved, "avg_days_late": avg_late,
            "late_count": late_count, "max_days_late": avg_late * 2,
        }

        try:
            df_input = predict_single(input_data)
            prob = pipeline.predict_proba(df_input)[0][1]
        except Exception as e:
            st.error(f"Помилка передбачення: {e}")
            st.stop()

        label, badge_class, emoji = risk_label(prob)
        score = get_credit_score(prob)
        threshold = metadata.get("optimal_threshold", 0.5)
        decision = "✅ СХВАЛИТИ" if prob < threshold else "❌ ВІДХИЛИТИ"

        st.markdown("---")
        st.markdown("### 📊 Результат оцінки")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"""
        <div class="metric-card">
          <div class="label">Ймовірність дефолту</div>
          <div class="value {'risk-high' if prob >= 0.6 else 'risk-medium' if prob >= 0.3 else 'risk-low'}">{prob*100:.1f}%</div>
          <div class="sub">поріг {threshold*100:.0f}%</div>
        </div>""", unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="metric-card">
          <div class="label">Кредитний скор</div>
          <div class="value">{score}</div>
          <div class="sub">300–850 шкала</div>
        </div>""", unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="metric-card">
          <div class="label">Рівень ризику</div>
          <div class="value {'risk-low' if label=='НИЗЬКИЙ' else 'risk-medium' if label=='СЕРЕДНІЙ' else 'risk-high'}">{emoji} {label}</div>
          <div class="sub">&nbsp;</div>
        </div>""", unsafe_allow_html=True)

        m4.markdown(f"""
        <div class="metric-card">
          <div class="label">Рекомендація</div>
          <div class="value" style="font-size:20px;">{decision}</div>
          <div class="sub">авто-рішення</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        col_gauge, col_bars = st.columns([1, 1])
        with col_gauge:
            st.plotly_chart(make_gauge(prob), use_container_width=True)

        with col_bars:
            st.markdown("**Ймовірність дефолту**")
            st.plotly_chart(make_prob_bar(prob), use_container_width=True)

            # Key ratios
            credit_income = credit / (income + 1)
            annuity_income = annuity * 12 / (income + 1)

            ratio_data = {
                "Кредит / Дохід": (credit_income, 3.0, "lower"),
                "Ануїтет / Річний дохід": (annuity_income, 0.5, "lower"),
                "Схвалення (попередні)": (
                    prev_approved / (prev_apps + 1), 0.5, "higher"),
                "EXT_SOURCE середнє": (
                    (ext1 + ext2 + ext3) / 3, 0.5, "higher"),
            }

            for name, (val, threshold_val, direction) in ratio_data.items():
                good = val <= threshold_val if direction == "lower" else val >= threshold_val
                icon = "🟢" if good else "🔴"
                st.markdown(f"{icon} **{name}:** `{val:.3f}`")

        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Рекомендації")

        recs = []
        if prob >= threshold:
            if ext1 < 0.4 or ext2 < 0.4 or ext3 < 0.4:
                recs.append(("⚠️", "Низький зовнішній скоринг. Запросити додаткові документи."))
            if credit / (income + 1) > 3:
                recs.append(("⚠️", "Високе боргове навантаження (кредит > 3× дохід)."))
            if late_count > 5:
                recs.append(("❌", "Значна кількість прострочених платежів."))
            if age < 27:
                recs.append(("ℹ️", "Молодий вік — підвищений ризик."))
            recs.append(("🔍", "Розглянути вимогу поручителя або застави."))
        else:
            recs.append(("✅", "Профіль клієнта відповідає критеріям схвалення."))
            if prob < 0.15:
                recs.append(("💎", "Відмінний профіль — розглянути VIP умови."))
            recs.append(("📋", "Стандартна перевірка документів."))

        for icon, text in recs:
            st.markdown(f"- {icon} {text}")


# ═══════════════════════════════════════════════
# PAGE 2: Portfolio Analysis
# ═══════════════════════════════════════════════
elif page == "📊 Аналіз портфеля":
    st.markdown("## 📊 Аналіз кредитного портфеля")

    if pipeline is None:
        st.error("❌ Модель не знайдена.")
        st.stop()

    st.info("Завантажте CSV файл з клієнтами для пакетного скорингу")

    uploaded_file = st.file_uploader(
        "Завантажте application_train.csv або будь-який CSV з ознаками клієнтів",
        type=["csv"],
        help="Файл повинен містити ті ж колонки, що й application_train"
    )

    if uploaded_file:
        with st.spinner("Обробка файлу..."):
            df_upload = pd.read_csv(uploaded_file)
            df_upload.columns = [c.lower() for c in df_upload.columns]

        st.success(f"✅ Завантажено: {df_upload.shape[0]:,} клієнтів")

        has_target = "target" in df_upload.columns

        try:
            # Try to get predictions
            from src.preprocessing import prepare_data as _prepare

            if has_target:
                X_port, y_port, _, _, prep_port = _prepare(df_upload)
                probs = pipeline.predict_proba(X_port)[:, 1]
                df_results = X_port.copy()
                df_results["probability"] = probs
                df_results["target"] = y_port.values
            else:
                drop_cols = ["sk_id_curr", "sk_id_bureau", "sk_id_prev", "index"]
                df_work = df_upload.drop(columns=[c for c in drop_cols if c in df_upload.columns])
                probs = pipeline.predict_proba(df_work)[:, 1]
                df_results = df_upload.copy()
                df_results["probability"] = probs
        except Exception as e:
            st.error(f"Помилка обробки: {e}")
            st.stop()

        threshold = metadata.get("optimal_threshold", 0.5)
        df_results["risk_level"] = pd.cut(
            df_results["probability"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Низький", "Середній", "Високий"]
        )
        df_results["decision"] = (df_results["probability"] < threshold).map(
            {True: "Схвалити", False: "Відхилити"}
        )
        df_results["credit_score"] = df_results["probability"].apply(get_credit_score)

        # KPIs
        st.markdown("### 📊 Ключові показники портфеля")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Всього клієнтів", f"{len(df_results):,}")
        k2.metric("Схвалити", f"{(df_results['decision']=='Схвалити').sum():,}",
                  f"{(df_results['decision']=='Схвалити').mean():.1%}")
        k3.metric("Середній скор", f"{df_results['credit_score'].mean():.0f}")
        k4.metric("Середня P(дефолту)", f"{df_results['probability'].mean():.1%}")
        if has_target:
            k5.metric("Фактичний дефолт", f"{df_results['target'].mean():.1%}")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            risk_counts = df_results["risk_level"].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={"Низький": "#22c55e", "Середній": "#f59e0b", "Високий": "#ef4444"},
                title="Розподіл за рівнем ризику"
            )
            fig_pie.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                                  font={"color": "#ccc"})
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_hist = px.histogram(
                df_results, x="probability", nbins=50,
                color_discrete_sequence=["#6366f1"],
                title="Розподіл ймовірностей дефолту"
            )
            fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#ef4444",
                               annotation_text=f"Поріг ({threshold:.2f})")
            fig_hist.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                                   font={"color": "#ccc"})
            st.plotly_chart(fig_hist, use_container_width=True)

        # Score distribution
        fig_score = px.histogram(
            df_results, x="credit_score", nbins=50,
            color_discrete_sequence=["#22c55e"],
            title="Розподіл кредитних скорів (300–850)"
        )
        fig_score.update_layout(paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                                 font={"color": "#ccc"})
        st.plotly_chart(fig_score, use_container_width=True)

        # Download results
        csv_out = df_results[["probability", "risk_level", "decision", "credit_score"]].to_csv(index=False)
        st.download_button(
            "⬇️ Завантажити результати CSV",
            data=csv_out,
            file_name="scoring_results.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════
# PAGE 3: Model Metrics
# ═══════════════════════════════════════════════
elif page == "📈 Метрики моделі":
    st.markdown("## 📈 Метрики та аналіз моделі")

    plots_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "plots"
    )

    if metadata:
        st.markdown("### 🏆 Результати Cross-Validation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CV ROC-AUC (mean)", f"{metadata.get('cv_auc_mean', 0):.4f}")
        m2.metric("CV ROC-AUC (std)", f"±{metadata.get('cv_auc_std', 0):.4f}")
        m3.metric("Test ROC-AUC", f"{metadata.get('test_auc', 0):.4f}")
        m4.metric("Avg Precision", f"{metadata.get('test_avg_precision', 0):.4f}")

    st.markdown("---")

    # Show plots if they exist
    plot_files = {
        "cv_results.png": "Результати Cross-Validation",
        "model_metrics.png": "ROC, PR, Confusion Matrix",
        "threshold_analysis.png": "Аналіз порогу",
        "feature_importance.png": "Важливість ознак",
        "score_analysis.png": "Аналіз скору по децилях",
        "shap_importance.png": "SHAP Importance",
        "shap_beeswarm.png": "SHAP Beeswarm",
    }

    if os.path.exists(plots_dir):
        for fname, title in plot_files.items():
            fpath = os.path.join(plots_dir, fname)
            if os.path.exists(fpath):
                st.markdown(f"#### {title}")
                st.image(fpath, use_column_width=True)
                st.markdown("---")
    else:
        st.info("📭 Графіки не знайдено. Запустіть `notebooks/model.ipynb` для генерації.")

    # Feature importance table
    if not importance_df.empty:
        st.markdown("### 📋 Таблиця важливості ознак")
        n_show = st.slider("Показати топ N ознак", 10, len(importance_df), 30)
        st.dataframe(
            importance_df.head(n_show).style.background_gradient(
                subset=["importance"], cmap="YlOrRd"
            ),
            use_container_width=True,
            height=400,
        )


# ═══════════════════════════════════════════════
# PAGE 4: About
# ═══════════════════════════════════════════════
elif page == "ℹ️ Про модель":
    st.markdown("## ℹ️ Про модель та проєкт")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
### 🏦 Home Credit Default Risk

Система кредитного скорингу для оцінки ймовірності дефолту
клієнтів Home Credit Group.

**Завдання:** бінарна класифікація — дефолт (1) / без дефолту (0)

**Метрика:** ROC-AUC

---

### 📊 Дані

| Таблиця | Опис |
|---------|------|
| `application_train` | Основна таблиця заявок |
| `bureau` | Кредитна історія (ЦБ) |
| `bureau_balance` | Баланси кредитів бюро |
| `previous_application` | Попередні заявки HC |
| `pos_cash_balance` | POS кредити та готівка |
| `installments_payments` | Платежі на виплат |
| `credit_card_balance` | Баланси кредитних карток |

---

### 🤖 Модель

**LightGBM Classifier**
- Gradient Boosting на деревах рішень
- Оптимізований для великих датасетів
- Підтримує категоріальні ознаки
- Висока швидкість навчання
""")

    with col2:
        st.markdown("""
### 🔧 Архітектура Pipeline

```
PostgreSQL DB
     │
     ▼
Feature Engineering (7 таблиць)
     │ ~80 ознак
     ▼
Preprocessing
 ├─ SimpleImputer (median)
 ├─ OrdinalEncoder
 └─ ColumnTransformer
     │
     ▼
LightGBM
 ├─ n_estimators: 1000
 ├─ learning_rate: 0.05
 ├─ num_leaves: 31
 └─ class_weight: balanced
     │
     ▼
5-fold StratifiedKFold CV
     │
     ▼
ROC-AUC ~0.76+
```

---

### 📦 Стек технологій

- **Database:** PostgreSQL + SQLAlchemy + pg8000
- **ML:** LightGBM, scikit-learn
- **Explainability:** SHAP
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Deploy:** Streamlit

---

### 🚀 Запуск

```bash
# 1. Ініціалізація БД
python init_db.py

# 2. Тренування (notebook)
jupyter notebook notebooks/model.ipynb

# 3. Запуск застосунку
streamlit run streamlit_app/app.py
```
""")

    if metadata:
        st.markdown("---")
        st.markdown("### 📊 Метадані моделі")
        st.json(metadata)