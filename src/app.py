"""
app.py — Home Credit Default Risk · Streamlit Dashboard

Запуск: streamlit run streamlit_app/app.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Home Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0f1117; }

.metric-card {
  background: linear-gradient(135deg,#1e1e2e 0%,#2a2a3e 100%);
  border: 1px solid #3a3a5e; border-radius: 12px;
  padding: 18px 20px; text-align: center;
}
.metric-card .label {
  color: #8888aa; font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;
}
.metric-card .value { color: #fff; font-size: 26px; font-weight: 700; line-height: 1.1; }
.metric-card .sub   { color: #6666aa; font-size: 11px; margin-top: 4px; }

.risk-low    { color: #22c55e !important; }
.risk-medium { color: #f59e0b !important; }
.risk-high   { color: #ef4444 !important; }

.section-header {
  color: #a0a0cc; font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 1.5px;
  margin: 18px 0 8px; border-bottom: 1px solid #2a2a4a; padding-bottom: 5px;
}
div[data-testid="stSidebar"] { background-color: #12121e; border-right: 1px solid #2a2a4a; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────
_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_DIR = os.path.join(_ROOT, "models")

# ── Load artefacts ────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    p = os.path.join(_MODELS_DIR, "credit_scoring_lgbm.pkl")
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data
def load_metadata() -> dict:
    p = os.path.join(_MODELS_DIR, "model_metadata.json")
    return json.load(open(p)) if os.path.exists(p) else {}

@st.cache_data
def load_importance() -> pd.DataFrame:
    p = os.path.join(_MODELS_DIR, "feature_importance.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

pipeline      = load_pipeline()
metadata      = load_metadata()
importance_df = load_importance()

# Колонки, які очікує модель (витягуємо з метаданих або з самого pipeline)
@st.cache_data
def get_model_feature_names() -> list[str]:
    """Повертає список ознак у тому порядку, як вчила модель."""
    # 1) спробувати metadata
    if metadata.get("feature_names"):
        return metadata["feature_names"]
    # 2) витягнути з preprocessor
    if pipeline is not None:
        try:
            prep = pipeline.named_steps["preprocessor"]
            num  = prep.transformers_[0][2]
            cat  = prep.transformers_[1][2]
            return list(num) + list(cat)
        except Exception:
            pass
    return []

MODEL_COLS = get_model_feature_names()


# ── Helpers ───────────────────────────────────────────────────────────
def risk_label(p: float):
    if p < 0.3:   return "НИЗЬКИЙ",  "🟢", "#22c55e"
    if p < 0.6:   return "СЕРЕДНІЙ", "🟡", "#f59e0b"
    return             "ВИСОКИЙ",  "🔴", "#ef4444"

def credit_score(p: float) -> int:
    return int(300 + (1 - p) * 550)

def gauge(p: float):
    sc = credit_score(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sc,
        title={"text": "Кредитний скор", "font": {"size": 16, "color": "#aaaacc"}},
        delta={"reference": 600, "decreasing": {"color": "#ef4444"}, "increasing": {"color": "#22c55e"}},
        gauge={
            "axis": {"range": [300, 850], "tickfont": {"color": "#999"}},
            "bar":  {"color": "#6366f1", "thickness": 0.28},
            "bgcolor": "#1a1a2e", "borderwidth": 1, "bordercolor": "#333",
            "steps": [
                {"range": [300, 500], "color": "#3b0a0a"},
                {"range": [500, 620], "color": "#3b2000"},
                {"range": [620, 720], "color": "#0a2e0a"},
                {"range": [720, 850], "color": "#063c06"},
            ],
        },
        number={"font": {"size": 38, "color": "#fff"}},
    ))
    fig.update_layout(paper_bgcolor="#1e1e2e", height=260,
                      margin=dict(t=55, b=10, l=15, r=15))
    return fig

def prob_bar(p: float):
    color = "#22c55e" if p < 0.3 else "#f59e0b" if p < 0.6 else "#ef4444"
    fig = go.Figure(go.Bar(
        x=[p * 100], y=[""], orientation="h",
        marker_color=color,
        text=[f"{p*100:.1f}%"], textposition="outside",
        textfont={"color": "#fff", "size": 18},
    ))
    fig.add_vline(x=30, line_dash="dash", line_color="#22c55e55")
    fig.add_vline(x=60, line_dash="dash", line_color="#ef444455")
    fig.update_layout(
        xaxis=dict(range=[0, 110], showgrid=False, tickfont={"color": "#666"}),
        yaxis=dict(showgrid=False),
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        height=90, margin=dict(t=5, b=5, l=5, r=65),
        font={"color": "#ccc"},
    )
    return fig


def build_input_row(data: dict) -> pd.DataFrame:
    """
    Будує один рядок з введених даних і вирівнює його під MODEL_COLS.
    Всі відсутні колонки заповнюються NaN — preprocessing сам імпутує медіаною.
    """
    age           = data.get("age", 35)
    yrs_empl      = data.get("years_employed", 5)
    days_employed = -yrs_empl * 365 if yrs_empl > 0 else 365243

    # Явно задаємо лише ті колонки, що має форма
    known = {
        # ── Ідентифікатор (потім дропається препроцесором) ──
        "sk_id_curr": 0,
        # ── Основні ознаки application ─────────────────────
        "name_contract_type":          data.get("contract_type", "Cash loans"),
        "code_gender":                 data.get("gender", "M"),
        "flag_own_car":                "Y" if data.get("own_car") else "N",
        "flag_own_realty":             "Y" if data.get("own_realty") else "N",
        "cnt_children":                data.get("children", 0),
        "amt_income_total":            data.get("income", 100000),
        "amt_credit":                  data.get("credit", 200000),
        "amt_annuity":                 data.get("annuity", 15000),
        "amt_goods_price":             data.get("goods_price", 180000),
        "name_income_type":            data.get("income_type", "Working"),
        "name_education_type":         data.get("education", "Secondary / secondary special"),
        "name_family_status":          data.get("family_status", "Married"),
        "name_housing_type":           data.get("housing", "House / apartment"),
        "region_population_relative":  0.02,
        "days_birth":                  -age * 365,
        "days_employed":               days_employed,
        "days_registration":           -5 * 365,
        "days_id_publish":             -5 * 365,
        "flag_mobil":                  1, "flag_emp_phone": 1,
        "flag_work_phone":             0, "flag_cont_mobile": 1,
        "flag_phone":                  0, "flag_email": 0,
        "occupation_type":             data.get("occupation", "Laborers"),
        "cnt_fam_members":             data.get("children", 0) + 2,
        "region_rating_client":        2,
        "region_rating_client_w_city": 2,
        "weekday_appr_process_start":  "WEDNESDAY",
        "hour_appr_process_start":     10,
        "reg_region_not_live_region":  0, "reg_region_not_work_region": 0,
        "live_region_not_work_region": 0,
        "reg_city_not_live_city":      0, "reg_city_not_work_city": 0,
        "live_city_not_work_city":     0,
        "organization_type":           data.get("org_type", "Business Entity Type 3"),
        "ext_source_1":                data.get("ext_source_1", 0.5),
        "ext_source_2":                data.get("ext_source_2", 0.5),
        "ext_source_3":                data.get("ext_source_3", 0.5),
        "obs_30_cnt_social_circle":    0, "def_30_cnt_social_circle": 0,
        "obs_60_cnt_social_circle":    0, "def_60_cnt_social_circle": 0,
        "days_last_phone_change":     -300,
        "flag_document_3":             1,
        "amt_req_credit_bureau_hour":  0, "amt_req_credit_bureau_day": 0,
        "amt_req_credit_bureau_week":  0, "amt_req_credit_bureau_mon": 1,
        "amt_req_credit_bureau_qrt":   2, "amt_req_credit_bureau_year": 5,
        # ── Агреговані ознаки (feature engineering) ────────
        "bureau_loan_count":           data.get("bureau_loans", 3),
        "bureau_active_loans":         1,
        "bureau_closed_loans":         2,
        "bureau_avg_days_credit":     -500,
        "bureau_avg_days_enddate":     100,
        "bureau_total_debt":           data.get("bureau_debt", 50000),
        "bureau_total_credit":         data.get("bureau_credit", 100000),
        "bureau_avg_overdue":          0,
        "bureau_max_overdue":          0,
        "bureau_sum_overdue":          0,
        "bureau_debt_ratio":           data.get("bureau_debt", 50000) / (data.get("bureau_credit", 100000) + 1),
        "bb_months_count":             12,
        "bb_avg_months_balance":      -6,
        "bb_dpd_count":                0,
        "bb_overdue_count":            0,
        "prev_app_count":              data.get("prev_apps", 2),
        "prev_app_approved":           data.get("prev_approved", 1),
        "prev_app_refused":            data.get("prev_refused", 0),
        "prev_avg_credit":             150000,
        "prev_avg_annuity":            12000,
        "prev_avg_application":        150000,
        "prev_avg_down_payment":       0,
        "prev_max_credit":             200000,
        "prev_avg_days_decision":     -200,
        "prev_avg_days_last_due":     -100,
        "prev_approval_rate":          data.get("prev_approved", 1) / (data.get("prev_apps", 2) + 1),
        "pos_months_count":            12,
        "pos_avg_instalment":          24,
        "pos_avg_future":              12,
        "pos_completed":               1,
        "pos_active":                  1,
        "pos_max_dpd":                 0,
        "pos_avg_dpd":                 0.0,
        "pos_max_dpd_def":             0,
        "inst_count":                  24,
        "inst_avg_amount":             10000,
        "inst_total_payment":          240000,
        "inst_avg_payment_diff":       0.0,
        "inst_max_payment_diff":       0.0,
        "inst_avg_days_late":          data.get("avg_days_late", 0),
        "inst_max_days_late":          data.get("max_days_late", 0),
        "inst_late_count":             data.get("late_count", 0),
        "inst_late_rate":              data.get("late_count", 0) / 25,
        "cc_months_count":             12,
        "cc_avg_balance":              data.get("cc_balance", 5000),
        "cc_max_balance":              data.get("cc_balance", 5000) * 2,
        "cc_avg_credit_limit":         data.get("cc_limit", 20000),
        "cc_avg_drawings":             1000.0,
        "cc_avg_payment":              2000.0,
        "cc_avg_receivable":           4000.0,
        "cc_max_dpd":                  0,
        "cc_avg_dpd":                  0.0,
        "cc_utilization":              data.get("cc_balance", 5000) / (data.get("cc_limit", 20000) + 1),
    }

    # ── Вирівнюємо під MODEL_COLS ──────────────────────────────────────
    # Якщо MODEL_COLS відомий — створюємо рядок з усіма потрібними колонками.
    # Будь-яка колонка, якої немає в known, стає NaN → імпутується медіаною.
    if MODEL_COLS:
        row = {col: known.get(col, np.nan) for col in MODEL_COLS}
    else:
        # Fallback: передаємо все що є — pipeline сам дропне зайве через remainder='drop'
        row = known

    return pd.DataFrame([row])


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏦 Credit Risk")
    st.markdown("### Home Credit Default")
    st.markdown("---")

    page = st.radio(
        "nav", ["🔮 Скоринг клієнта", "📊 Аналіз портфеля", "📈 Метрики моделі", "ℹ️ Про проєкт"],
        label_visibility="hidden",
    )

    st.markdown("---")
    if pipeline is not None and metadata:
        st.markdown("**Модель:** LightGBM")
        st.markdown(f"**CV AUC:** `{metadata.get('cv_auc_mean', 0):.4f}`")
        st.markdown(f"**Ознак:** `{metadata.get('n_features', '?')}`")
    elif pipeline is None:
        st.error("Модель не знайдена")
        st.markdown("Запустіть `notebooks/model.ipynb`")

    st.markdown("---")
    st.caption("LightGBM · SHAP · Streamlit")
    st.caption("Home Credit Default Risk")


# ══════════════════════════════════════════════
# PAGE 1 — Скоринг клієнта
# ══════════════════════════════════════════════
if page == "🔮 Скоринг клієнта":
    st.markdown("## 🔮 Скоринг клієнта")
    st.markdown("Заповніть параметри — модель розрахує ймовірність дефолту та кредитний скор.")

    if pipeline is None:
        st.error("❌ Модель не знайдена. Спочатку запустіть `notebooks/model.ipynb`")
        st.stop()

    with st.form("scoring_form"):
        st.markdown('<div class="section-header">👤 Персональні дані</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        age        = c1.number_input("Вік", 18, 75, 35)
        gender     = c2.selectbox("Стать", ["M", "F"])
        children   = c3.number_input("Дітей", 0, 10, 0)
        family     = c4.selectbox("Сімейний стан", [
            "Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
        c1, c2 = st.columns(2)
        own_car    = c1.checkbox("🚗 Має автомобіль")
        own_realty = c2.checkbox("🏠 Має нерухомість", value=True)

        st.markdown('<div class="section-header">💼 Зайнятість</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        income_type    = c1.selectbox("Тип доходу", [
            "Working", "Commercial associate", "Pensioner",
            "State servant", "Businessman", "Student", "Maternity leave"])
        education      = c2.selectbox("Освіта", [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"])
        occupation     = c3.selectbox("Професія", [
            "Laborers", "Core staff", "Managers", "Drivers",
            "High skill tech staff", "Accountants", "Medicine staff",
            "Security staff", "Cooking staff", "Cleaning staff",
            "Private service staff", "Low-skill Laborers",
            "HR staff", "Realty agents", "Secretaries", "IT staff",
            "Waiters/barmen staff"])
        years_employed = c4.number_input("Стаж (рр)", 0, 50, 5)

        st.markdown('<div class="section-header">💰 Фінансові показники</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        income  = c1.number_input("Дохід (грн)",        10_000, 5_000_000, 100_000, step=10_000)
        credit  = c2.number_input("Сума кредиту (грн)", 10_000, 5_000_000, 200_000, step=10_000)
        annuity = c3.number_input("Ануїтет/місяць",      1_000,   200_000,  15_000, step=1_000)
        goods   = c4.number_input("Вартість товару",          0, 5_000_000, 180_000, step=10_000)

        st.markdown('<div class="section-header">🏛️ Зовнішній скоринг (EXT_SOURCE)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        ext1 = c1.slider("EXT_SOURCE_1", 0.0, 1.0, 0.50, 0.01, help="Зовнішній скоринг 1 (0=поганий, 1=відмінний)")
        ext2 = c2.slider("EXT_SOURCE_2", 0.0, 1.0, 0.50, 0.01, help="Зовнішній скоринг 2")
        ext3 = c3.slider("EXT_SOURCE_3", 0.0, 1.0, 0.50, 0.01, help="Зовнішній скоринг 3")

        st.markdown('<div class="section-header">📋 Кредитна історія</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        prev_apps     = c1.number_input("Попередніх заявок", 0, 50, 2)
        prev_approved = c2.number_input("Схвалено",          0, 50, 1)
        avg_late      = c3.number_input("Середн. прострочення (дн)", 0, 180, 0)
        late_count    = c4.number_input("К-сть прострочених",        0, 200, 0)

        submitted = st.form_submit_button("🎯 Розрахувати скор", use_container_width=True, type="primary")

    if submitted:
        input_data = {
            "age": age, "gender": gender, "children": children,
            "family_status": family, "own_car": own_car, "own_realty": own_realty,
            "income_type": income_type, "education": education,
            "occupation": occupation, "years_employed": years_employed,
            "income": income, "credit": credit, "annuity": annuity,
            "goods_price": goods,
            "ext_source_1": ext1, "ext_source_2": ext2, "ext_source_3": ext3,
            "prev_apps": prev_apps, "prev_approved": prev_approved,
            "avg_days_late": avg_late, "late_count": late_count,
            "max_days_late": avg_late * 2,
        }

        with st.spinner("Розрахунок..."):
            try:
                df_row = build_input_row(input_data)
                prob   = float(pipeline.predict_proba(df_row)[0][1])
            except Exception as e:
                st.error(f"Помилка передбачення: {e}")
                st.stop()

        label, emoji, color = risk_label(prob)
        sc        = credit_score(prob)
        threshold = metadata.get("optimal_threshold", 0.5)
        decision  = "✅ СХВАЛИТИ" if prob < threshold else "❌ ВІДХИЛИТИ"

        st.markdown("---")
        st.markdown("### 📊 Результат оцінки")

        m1, m2, m3, m4 = st.columns(4)
        risk_cls = "risk-low" if prob < 0.3 else "risk-medium" if prob < 0.6 else "risk-high"

        m1.markdown(f"""<div class="metric-card">
          <div class="label">Ймовірність дефолту</div>
          <div class="value {risk_cls}">{prob*100:.1f}%</div>
          <div class="sub">поріг {threshold*100:.0f}%</div>
        </div>""", unsafe_allow_html=True)

        m2.markdown(f"""<div class="metric-card">
          <div class="label">Кредитний скор</div>
          <div class="value">{sc}</div>
          <div class="sub">шкала 300–850</div>
        </div>""", unsafe_allow_html=True)

        m3.markdown(f"""<div class="metric-card">
          <div class="label">Рівень ризику</div>
          <div class="value {risk_cls}">{emoji} {label}</div>
          <div class="sub">&nbsp;</div>
        </div>""", unsafe_allow_html=True)

        m4.markdown(f"""<div class="metric-card">
          <div class="label">Рекомендація</div>
          <div class="value" style="font-size:18px;">{decision}</div>
          <div class="sub">авто-рішення</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        col_g, col_r = st.columns(2)
        with col_g:
            st.plotly_chart(gauge(prob), use_container_width=True)
        with col_r:
            st.markdown("**Ймовірність дефолту**")
            st.plotly_chart(prob_bar(prob), use_container_width=True)
            st.markdown("---")
            credit_income  = credit  / (income + 1)
            annuity_annual = annuity * 12 / (income + 1)
            ext_avg        = (ext1 + ext2 + ext3) / 3
            approval_rate  = prev_approved / (prev_apps + 1)

            checks = [
                ("Кредит / Дохід",         credit_income,  3.0,  "lower"),
                ("Ануїтет / Річн. дохід",  annuity_annual, 0.5,  "lower"),
                ("EXT_SOURCE (середнє)",    ext_avg,        0.5,  "higher"),
                ("Рівень схвалень (попер)", approval_rate,  0.5,  "higher"),
            ]
            for name, val, thr, direction in checks:
                ok   = val <= thr if direction == "lower" else val >= thr
                icon = "🟢" if ok else "🔴"
                st.markdown(f"{icon} **{name}:** `{val:.3f}`")

        st.markdown("---")
        st.markdown("### 💡 Рекомендації")
        recs = []
        if prob >= threshold:
            if ext_avg < 0.4:
                recs.append(("⚠️", "Низький зовнішній скоринг — запросити додаткові документи."))
            if credit_income > 3:
                recs.append(("⚠️", f"Боргове навантаження {credit_income:.1f}× дохід (норма < 3×)."))
            if late_count > 5:
                recs.append(("❌", f"{late_count} прострочених платежів — суттєвий ризик."))
            if age < 27:
                recs.append(("ℹ️", "Молодий вік — статистично вищий ризик дефолту."))
            if income_type == "Pensioner":
                recs.append(("ℹ️", "Пенсіонер — перевірити стабільність доходу."))
            recs.append(("🔍", "Розглянути поручителя або додаткову заставу."))
        else:
            recs.append(("✅", "Профіль клієнта відповідає критеріям схвалення."))
            if prob < 0.1:
                recs.append(("💎", "Відмінний профіль — клієнт категорії VIP."))
            recs.append(("📋", "Стандартна верифікація документів."))

        for icon, text in recs:
            st.markdown(f"- {icon} {text}")


# ══════════════════════════════════════════════
# PAGE 2 — Аналіз портфеля
# ══════════════════════════════════════════════
elif page == "📊 Аналіз портфеля":
    st.markdown("## 📊 Пакетний скоринг портфеля")
    st.info("Завантажте CSV з клієнтами для пакетного скорингу.", icon="📁")

    if pipeline is None:
        st.error("❌ Модель не знайдена.")
        st.stop()

    uploaded = st.file_uploader("CSV файл клієнтів", type=["csv"])
    if not uploaded:
        st.stop()

    with st.spinner("Читання файлу..."):
        df_up = pd.read_csv(uploaded)
        df_up.columns = [c.lower() for c in df_up.columns]

    st.success(f"✅ Завантажено: {df_up.shape[0]:,} клієнтів, {df_up.shape[1]} колонок")

    has_target = "target" in df_up.columns

    with st.spinner("Скоринг..."):
        try:
            from src.preprocessing import prepare_data as _prep
            if has_target:
                X_p, y_p, _, _, _ = _prep(df_up)
                probs = pipeline.predict_proba(X_p)[:, 1]
                df_res = pd.DataFrame({"probability": probs, "target": y_p.values})
            else:
                drop_c = ["sk_id_curr", "sk_id_bureau", "sk_id_prev", "index"]
                df_w   = df_up.drop(columns=[c for c in drop_c if c in df_up.columns])
                probs  = pipeline.predict_proba(df_w)[:, 1]
                df_res = pd.DataFrame({"probability": probs})
        except Exception as e:
            st.error(f"Помилка: {e}")
            st.stop()

    thr = metadata.get("optimal_threshold", 0.5)
    df_res["risk_level"]  = pd.cut(df_res["probability"], [0,.3,.6,1.0],
                                    labels=["Низький","Середній","Високий"])
    df_res["decision"]    = (df_res["probability"] < thr).map({True:"Схвалити", False:"Відхилити"})
    df_res["credit_score"]= df_res["probability"].apply(credit_score)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Клієнтів",          f"{len(df_res):,}")
    k2.metric("Схвалити",          f"{(df_res.decision=='Схвалити').sum():,}",
              f"{(df_res.decision=='Схвалити').mean():.1%}")
    k3.metric("Середній скор",     f"{df_res.credit_score.mean():.0f}")
    k4.metric("Середня P(дефолту)",f"{df_res.probability.mean():.1%}")
    if has_target:
        k5.metric("Факт. дефолт", f"{df_res.target.mean():.1%}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        rc = df_res.risk_level.value_counts()
        fig = px.pie(values=rc.values, names=rc.index,
                     color=rc.index,
                     color_discrete_map={"Низький":"#22c55e","Середній":"#f59e0b","Високий":"#ef4444"},
                     title="Розподіл за рівнем ризику")
        fig.update_layout(paper_bgcolor="#1e1e2e", font={"color":"#ccc"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df_res, x="probability", nbins=50,
                           color_discrete_sequence=["#6366f1"],
                           title="Розподіл ймовірностей дефолту")
        fig.add_vline(x=thr, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Поріг {thr:.2f}")
        fig.update_layout(paper_bgcolor="#1e1e2e", font={"color":"#ccc"})
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df_res, x="credit_score", nbins=50,
                       color_discrete_sequence=["#22c55e"],
                       title="Розподіл кредитних скорів (300–850)")
    fig.update_layout(paper_bgcolor="#1e1e2e", font={"color":"#ccc"})
    st.plotly_chart(fig, use_container_width=True)

    csv_out = df_res[["probability","risk_level","decision","credit_score"]].to_csv(index=False)
    st.download_button("⬇️ Завантажити результати CSV", data=csv_out,
                       file_name="scoring_results.csv", mime="text/csv")


# ══════════════════════════════════════════════
# PAGE 3 — Метрики моделі
# ══════════════════════════════════════════════
elif page == "📈 Метрики моделі":
    st.markdown("## 📈 Метрики та аналіз моделі")

    if metadata:
        st.markdown("### 🏆 Результати Cross-Validation")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("CV ROC-AUC (mean)", f"{metadata.get('cv_auc_mean',0):.4f}")
        m2.metric("CV ROC-AUC (std)",  f"±{metadata.get('cv_auc_std',0):.4f}")
        m3.metric("Ознак у моделі",    metadata.get("n_features","?"))
        m4.metric("Поріг рішення",     f"{metadata.get('optimal_threshold',0.5):.2f}")

    st.markdown("---")

    plots_dir = os.path.join(_MODELS_DIR, "plots")
    plot_files = [
        ("cv_results.png",         "📊 Результати Cross-Validation"),
        ("model_metrics.png",      "📈 ROC, PR-крива, Confusion Matrix"),
        ("threshold_analysis.png", "🎯 Аналіз порогу класифікації"),
        ("feature_importance.png", "🔑 Важливість ознак"),
        ("score_analysis.png",     "📉 Аналіз скору по децилях"),
        ("shap_importance.png",    "🧠 SHAP Feature Importance"),
        ("shap_beeswarm.png",      "🐝 SHAP Beeswarm Plot"),
    ]

    if os.path.exists(plots_dir):
        found = False
        for fname, title in plot_files:
            fpath = os.path.join(plots_dir, fname)
            if os.path.exists(fpath):
                st.markdown(f"#### {title}")
                st.image(fpath, use_column_width=True)
                st.markdown("---")
                found = True
        if not found:
            st.info("📭 Графіки не знайдено — запустіть `notebooks/model.ipynb`.")
    else:
        st.info("📭 Папка `models/plots/` відсутня — запустіть `notebooks/model.ipynb`.")

    if not importance_df.empty:
        st.markdown("### 📋 Таблиця важливості ознак")
        n = st.slider("Топ N ознак", 10, min(len(importance_df), 100), 30)
        st.dataframe(
            importance_df.head(n).style.background_gradient(subset=["importance"], cmap="YlOrRd"),
            use_container_width=True, height=400,
        )


# ══════════════════════════════════════════════
# PAGE 4 — Про проєкт
# ══════════════════════════════════════════════
elif page == "ℹ️ Про проєкт":
    st.markdown("## ℹ️ Home Credit Default Risk")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### Задача
Бінарна класифікація: передбачити ймовірність дефолту позичальника.
Метрика оцінки — **ROC-AUC**.

### Дані
| Таблиця | Рядків |
|---------|--------|
| `application_train` | 307 511 |
| `bureau` | 1 716 428 |
| `bureau_balance` | 27 299 925 |
| `previous_application` | 1 670 214 |
| `pos_cash_balance` | 10 001 358 |
| `installments_payments` | 13 605 401 |
| `credit_card_balance` | 3 840 312 |

### ML Pipeline
```
PostgreSQL → parquet кеш (1 раз)
    ↓
Feature Engineering  (~80 ознак з 7 таблиць)
    ↓
sklearn Pipeline
  ├─ SimpleImputer (median)
  └─ OrdinalEncoder
    ↓
LightGBM Classifier
  ├─ n_estimators: 1000
  ├─ learning_rate: 0.05
  ├─ num_leaves: 31
  └─ class_weight: balanced
    ↓
5-fold StratifiedKFold CV → ROC-AUC ~0.76+
```
""")
    with col2:
        st.markdown("""
### Стек технологій

| Шар | Технологія |
|-----|-----------|
| Database | PostgreSQL + SQLAlchemy |
| Cache | Apache Parquet + pyarrow |
| ML | LightGBM + scikit-learn |
| Explainability | SHAP |
| Visualization | Plotly + Matplotlib |
| Deploy | Streamlit |

### Запуск проєкту
```bash
# 1. Встановити залежності
pip install -r requirements.txt

# 2. Дамп PostgreSQL → parquet (один раз)
python dump_to_parquet.py

# 3. EDA
jupyter notebook notebooks/EDA.ipynb

# 4. Тренування
jupyter notebook notebooks/model.ipynb

# 5. Streamlit
streamlit run streamlit_app/app.py
```

### Структура проєкту
```
├── data/parquet/        ← локальний кеш
├── models/              ← pkl, json, csv, plots
├── notebooks/           ← EDA.ipynb, model.ipynb
├── src/                 ← db, features, preprocessing, train
├── streamlit_app/       ← app.py
└── dump_to_parquet.py
```
""")

    if metadata:
        st.markdown("---")
        st.markdown("### 🔍 Метадані поточної моделі")
        # Показуємо все крім великого списку feature_names
        meta_display = {k: v for k, v in metadata.items()
                        if k not in ("feature_names",)}
        st.json(meta_display)