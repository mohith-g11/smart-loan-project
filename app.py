import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="Smart Loan Risk Analyzer", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3, h4, h5 { color: #58a6ff !important; font-family: 'Inter', sans-serif; }
    .metric-card {
        background-color: #21262d; border: 1px solid #30363d;
        border-radius: 6px; padding: 15px; text-align: center; margin-bottom: 10px;
    }
    .grade-badge { font-size: 28px; font-weight: bold; color: #58a6ff; }
    .card-header {
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 8px 8px 0 0; padding: 12px 20px; margin-bottom: 0;
    }
    div.stButton > button:first-child {
        background-color: #1f6feb !important; color: white !important;
        border-radius: 6px; width: 100%; border: none; font-weight: bold; height: 45px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INIT ---
if 'computed' not in st.session_state:
    st.session_state.computed = False
if 'gender_val' not in st.session_state:
    st.session_state.gender_val = 1

@st.cache_resource
def load_credit_model():
    return joblib.load('loan_model_pro.pkl')

model = load_credit_model()

st.title("🛡️ Smart Loan Risk Analyzer")
st.caption("Predicts loan approval chances using a real-world dataset of 51,000+ applicants.")
st.markdown("---")

app_mode = st.radio("⚡ Select Dashboard Mode:",
                    ["Single Applicant Analysis & Risk Simulator", "Compare Two Applicants Side-by-Side"],
                    horizontal=True)

# --- SHARED HELPER ---
def calculate_metrics_and_grades(inc, age, enq3m, tot_enq, delinq, std_acc, loan_amt, gender_num):
    if age < 18:
        return None

    payload = pd.DataFrame([{
        'Income': inc, 'Age': age, 'Recent_Enquiries_3M': enq3m,
        'Total_Enquiries_Ever': tot_enq, 'Total_Delinquencies': delinq,
        'Clean_Standard_Accounts': std_acc, 'Gender': gender_num
    }])
    payload = payload[['Income', 'Age', 'Recent_Enquiries_3M', 'Total_Enquiries_Ever',
                        'Total_Delinquencies', 'Clean_Standard_Accounts', 'Gender']]

    base_prob = model.predict_proba(payload)[0][1] * 100

    max_allowable = inc * 10
    overleveraged = loan_amt > max_allowable
    if overleveraged:
        base_prob = max(base_prob - 45.0, 5.0)

    r = 11 / (12 * 100)
    n = 5 * 12
    emi = loan_amt * (r * (1 + r)**n) / (((1 + r)**n) - 1) if loan_amt > 0 else 0
    f_ratio = (emi / inc) * 100 if inc > 0 else 100

    g_income  = 'A' if inc >= 80000 else 'B' if inc >= 45000 else 'C' if inc >= 25000 else 'D'
    g_history = 'A' if delinq == 0 else 'B' if delinq == 1 else 'C' if delinq <= 3 else 'F'
    g_hunger  = 'A' if enq3m <= 1 else 'B' if enq3m <= 3 else 'C' if enq3m <= 6 else 'D'
    g_health  = 'A' if std_acc >= 8 else 'B' if std_acc >= 4 else 'C' if std_acc >= 1 else 'D'

    return {
        "prob": base_prob, "emi": emi, "foir": f_ratio,
        "overleveraged": overleveraged, "max_allowable": max_allowable,
        "grades": [g_income, g_history, g_hunger, g_health], "payload": payload
    }

# --- COLOUR BAR HELPER (matplotlib-based, never broken by Streamlit HTML sanitizer) ---
def draw_gauge_bar(percent, bar_color):
    fig, ax = plt.subplots(figsize=(7, 0.55))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#2d3139')
    ax.barh(0, 100, color='#2d3139', height=0.5, left=0)
    ax.barh(0, percent, color=bar_color, height=0.5, left=0)
    ax.text(1,  0, "🔴 High Risk", va='center', ha='left',  color='#ff4b4b', fontsize=8, fontweight='bold')
    ax.text(99, 0, "🟢 Safe Zone", va='center', ha='right', color='#09ab3b', fontsize=8, fontweight='bold')
    ax.text(50, 0, f"{percent:.1f}%", va='center', ha='center', color='white', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.axis('off')
    plt.tight_layout(pad=0.1)
    return fig

# ==========================================
# MODE 1: SINGLE APPLICANT
# ==========================================
if app_mode == "Single Applicant Analysis & Risk Simulator":

    st.markdown("### Profile Input Parameters")
    col_left, col_right = st.columns(2)

    # BUG #2 FIX: Card headers are standalone markdown blocks.
    # Native widgets are placed directly in columns — NOT wrapped in open/close div tags.
    with col_left:
        st.markdown('<div class="card-header"><h4>📋 Loan Details</h4></div>', unsafe_allow_html=True)
        req_loan     = st.number_input("Requested Loan Amount (₹)", min_value=10000, max_value=10000000, value=300000, step=25000)
        st.caption("Standard safety policies cap loan limits at 10× your monthly income.")
        income_in    = st.number_input("Net Monthly Income (₹)", min_value=10000, max_value=1000000, value=45000, step=1000)
        gender_input = st.radio("Applicant Gender", options=["Male", "Female"], horizontal=True)
        age_method   = st.radio("Age Input Method", options=["Enter Age Directly", "Use Date of Birth"], horizontal=True)

        if age_method == "Use Date of Birth":
            dob   = st.date_input("Date of Birth", value=date(1998, 1, 1),
                                  min_value=date(1940, 1, 1), max_value=date(2008, 12, 31))
            today = date.today()
            age_in = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        else:
            age_in = st.number_input("Applicant Age", min_value=18, max_value=85, value=28, step=1)

    with col_right:
        st.markdown('<div class="card-header" style="border-left: 4px solid #a275ff;"><h4>📊 Credit History</h4></div>', unsafe_allow_html=True)
        clean_in   = st.number_input("Number of Clean Active Loans", min_value=0, max_value=100, value=4)
        delinq_in  = st.number_input("Times Paid EMI Late", min_value=0, max_value=50, value=0)
        enq3m_in   = st.number_input("Loan Applications (Last 3 Months)", min_value=0, max_value=30, value=0)
        tot_enq_in = st.number_input("Total Loan Applications Ever", min_value=0, max_value=100, value=5)

    st.write("")
    if st.button("🔥 Run Credit Assessment Engine"):
        # BUG #3 FIX: Reset computed on every validation failure so stale results don't persist
        if clean_in > tot_enq_in:
            st.session_state.computed = False
            st.error("⚠️ Data Entry Error: Clean Active Loans cannot exceed Total Loan Applications Ever.")
        elif enq3m_in > tot_enq_in:
            st.session_state.computed = False
            st.error("⚠️ Data Entry Error: Recent Applications (Last 3 Months) cannot exceed Total Loan Applications Ever.")
        elif age_in < 18:
            st.session_state.computed = False
            st.error("⚠️ Policy Rejection: Applicant must be 18 years or older.")
        else:
            # BUG #4 FIX: Store gender_val explicitly in session state on button click
            st.session_state.gender_val = 1 if gender_input == "Male" else 0
            st.session_state.computed = True

    if st.session_state.computed:
        # Use stored gender from session state — robust across slider rerenders
        gender_val = st.session_state.gender_val
        res = calculate_metrics_and_grades(
            income_in, age_in, enq3m_in, tot_enq_in,
            delinq_in, clean_in, req_loan, gender_val
        )

        st.markdown("---")
        st.markdown("### 📊 Assessment Summary")

        p = res["prob"]

        if p >= 65 and not res["overleveraged"]:
            status_title, bar_color, pill_style = "APPROVED", "#09ab3b", "background-color:#09ab3b;"
        elif p <= 40 or res["overleveraged"]:
            status_title, bar_color, pill_style = "REJECTED", "#ff4b4b", "background-color:#ff4b4b;"
        else:
            status_title, bar_color, pill_style = "MANUAL VERIFICATION REQUIRED", "#faca2b", "background-color:#faca2b; color:#0d1117;"

        st.markdown(
            f'<div style="text-align:center; margin-bottom:25px;">'
            f'<span style="{pill_style} color:white; padding:12px 40px; border-radius:50px; '
            f'font-size:24px; font-weight:bold; letter-spacing:1px;">DECISION: {status_title}</span></div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><h5>Approval Probability</h5><h2>{p:.2f}%</h2>'
                        f'<p style="font-size:0.8em; color:#8b949e;">Model Confidence Score</p></div>',
                        unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><h5>Estimated Monthly EMI</h5><h2>₹{res["emi"]:,.2f}</h2>'
                        f'<p style="font-size:0.8em; color:#8b949e;">5-Year Term at 11% p.a.</p></div>',
                        unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><h5>Maximum Safe Loan Limit</h5><h2>₹{res["max_allowable"]:,}</h2>'
                        f'<p style="font-size:0.8em; color:#8b949e;">10× Monthly Income Cap</p></div>',
                        unsafe_allow_html=True)

        if res["foir"] > 40:
            st.error(f"⚠️ **High Debt Obligation Risk:** EMI consumes {res['foir']:.1f}% of monthly income — exceeds safe 40% RBI threshold.")
        else:
            st.success(f"✅ **Safe Leverage:** EMI consumes a healthy {res['foir']:.1f}% of monthly income.")

        # BUG #7 FIX: Coloured matplotlib gauge bar instead of plain st.progress()
        st.write("")
        st.markdown("**Risk Score Gauge**")
        fig_gauge = draw_gauge_bar(p, bar_color)
        st.pyplot(fig_gauge)
        plt.close(fig_gauge)

        # Credit Health Scorecard
        st.markdown("#### 🎫 Credit Health Scorecard")
        g_cols  = st.columns(4)
        g_labels = ["Income Strength", "Repayment History", "Application Velocity", "Account Health"]
        for i, col in enumerate(g_cols):
            with col:
                st.markdown(
                    f'<div class="metric-card" style="border-top: 3px solid #58a6ff;">'
                    f'<p style="font-size:0.9em; margin-bottom:2px; font-weight:600;">{g_labels[i]}</p>'
                    f'<span class="grade-badge">{res["grades"][i]}</span></div>',
                    unsafe_allow_html=True
                )

        # What-If Simulator
        st.write("")
        with st.expander("🔮 Interactive What-If Scenario Simulator", expanded=True):
            st.info("Adjust sliders to instantly simulate alternative outcomes without clearing your main results.")
            sim_loan   = st.slider("Simulated Loan Request (₹)", min_value=10000, max_value=2000000, value=int(req_loan), step=10000)
            sim_delinq = st.slider("Simulated Late Payments Count", min_value=0, max_value=20, value=int(delinq_in))
            sim_res    = calculate_metrics_and_grades(income_in, age_in, enq3m_in, tot_enq_in,
                                                      sim_delinq, clean_in, sim_loan, gender_val)
            st.metric(label="Re-calculated Approval Probability",
                      value=f"{sim_res['prob']:.2f}%",
                      delta=f"{sim_res['prob'] - p:.2f}% vs baseline")

        # 5-Year Trend
        with st.expander("📈 5-Year Income Growth Risk Projection"):
            st.caption("Approval probability trajectory assuming 10% annual income growth.")
            years  = ["Year 1", "Year 2 (+10%)", "Year 3 (+20%)", "Year 4 (+30%)", "Year 5 (+40%)"]
            trend  = []
            for idx in range(5):
                t_inc = income_in * (1.10 ** idx)
                t_res = calculate_metrics_and_grades(t_inc, age_in + idx, enq3m_in,
                                                     tot_enq_in, delinq_in, clean_in,
                                                     req_loan, gender_val)
                trend.append(t_res["prob"])
            chart_df = pd.DataFrame({"Year": years, "Approval Probability (%)": trend})
            st.line_chart(data=chart_df, x="Year", y="Approval Probability (%)", use_container_width=True)

        # SHAP — BUG #6 FIX: plt.close(fig) added
        with st.expander("🔍 Advanced Model Insights (SHAP Feature Weights)"):
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(res["payload"])
            fig, ax    = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#161b22')
            ax.set_facecolor('#161b22')
            shap.waterfall_plot(shap.Explanation(
                values=shap_vals[0], base_values=explainer.expected_value,
                data=res["payload"].iloc[0], feature_names=res["payload"].columns.tolist()
            ), show=False)
            st.pyplot(fig)
            plt.close(fig)  # BUG #6 FIX

# ==========================================
# MODE 2: SIDE-BY-SIDE COMPARISON
# ==========================================
else:
    st.session_state.computed = False
    st.markdown("### 🔀 Applicant Comparison Mode")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card-header" style="border-left: 4px solid #58a6ff;"><h4>👤 Applicant Alpha</h4></div>', unsafe_allow_html=True)
        l_a    = st.number_input("Alpha: Loan Amount (₹)",             value=400000, step=25000, key="la")
        i_a    = st.number_input("Alpha: Monthly Income (₹)",          value=60000,  step=2000,  key="ia")
        age_a  = st.number_input("Alpha: Age",                         value=32,                 key="agea")
        std_a  = st.number_input("Alpha: Clean Active Loans",          value=8,                  key="stda")
        del_a  = st.number_input("Alpha: Times Paid EMI Late",         value=0,                  key="dela")
        enq3_a = st.number_input("Alpha: Applications (Last 3 Months)",value=0,                  key="e3a")
        tot_a  = st.number_input("Alpha: Total Applications Ever",     value=8,                  key="tota")

    with col2:
        st.markdown('<div class="card-header" style="border-left: 4px solid #ff7b72;"><h4>👤 Applicant Beta</h4></div>', unsafe_allow_html=True)
        l_b    = st.number_input("Beta: Loan Amount (₹)",              value=400000, step=25000, key="lb")
        i_b    = st.number_input("Beta: Monthly Income (₹)",           value=35000,  step=2000,  key="ib")
        age_b  = st.number_input("Beta: Age",                          value=24,                 key="ageb")
        std_b  = st.number_input("Beta: Clean Active Loans",           value=2,                  key="stdb")
        del_b  = st.number_input("Beta: Times Paid EMI Late",          value=4,                  key="delb")
        enq3_b = st.number_input("Beta: Applications (Last 3 Months)", value=5,                  key="e3b")
        tot_b  = st.number_input("Beta: Total Applications Ever",      value=12,                 key="totb")

    if st.button("🔥 Run Side-by-Side Comparative Underwriting Engine"):
        res_a = calculate_metrics_and_grades(i_a, age_a, enq3_a, tot_a, del_a, std_a, l_a, 1)
        res_b = calculate_metrics_and_grades(i_b, age_b, enq3_b, tot_b, del_b, std_b, l_b, 1)

        if res_a and res_b:
            st.markdown("### 📊 Comparative Analysis Results")
            o1, o2 = st.columns(2)

            with o1:
                st.markdown(
                    f'<div class="metric-card" style="border: 2px solid #58a6ff;">'
                    f'<h3>Alpha Score: {res_a["prob"]:.2f}%</h3>'
                    f'<p>Monthly EMI: ₹{res_a["emi"]:,.2f}<br>'
                    f'Income Used for EMI: {res_a["foir"]:.1f}%</p>'
                    f'<h4>Report Card: {" | ".join(res_a["grades"])}</h4></div>',
                    unsafe_allow_html=True
                )
                # BUG #1 FIX: shap.bar_plot replaced with shap.plots.bar via Explanation object
                explainer_a = shap.TreeExplainer(model)
                shap_vals_a = explainer_a.shap_values(res_a["payload"])
                fig_a, ax_a = plt.subplots(figsize=(5, 2.5))
                fig_a.patch.set_facecolor('#161b22')
                ax_a.set_facecolor('#161b22')
                shap_exp_a = shap.Explanation(
                    values=shap_vals_a[0],
                    base_values=explainer_a.expected_value,
                    data=res_a["payload"].iloc[0],
                    feature_names=res_a["payload"].columns.tolist()
                )
                shap.plots.bar(shap_exp_a, max_display=5, show=False)
                st.pyplot(fig_a)
                plt.close(fig_a)  # BUG #5 FIX

            with o2:
                st.markdown(
                    f'<div class="metric-card" style="border: 2px solid #ff7b72;">'
                    f'<h3>Beta Score: {res_b["prob"]:.2f}%</h3>'
                    f'<p>Monthly EMI: ₹{res_b["emi"]:,.2f}<br>'
                    f'Income Used for EMI: {res_b["foir"]:.1f}%</p>'
                    f'<h4>Report Card: {" | ".join(res_b["grades"])}</h4></div>',
                    unsafe_allow_html=True
                )
                # BUG #1 FIX: same fix for Beta
                explainer_b = shap.TreeExplainer(model)
                shap_vals_b = explainer_b.shap_values(res_b["payload"])
                fig_b, ax_b = plt.subplots(figsize=(5, 2.5))
                fig_b.patch.set_facecolor('#161b22')
                ax_b.set_facecolor('#161b22')
                shap_exp_b = shap.Explanation(
                    values=shap_vals_b[0],
                    base_values=explainer_b.expected_value,
                    data=res_b["payload"].iloc[0],
                    feature_names=res_b["payload"].columns.tolist()
                )
                shap.plots.bar(shap_exp_b, max_display=5, show=False)
                st.pyplot(fig_b)
                plt.close(fig_b)  # BUG #5 FIX