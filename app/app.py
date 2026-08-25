"""
US Personal Auto UW & Pricing - Streamlit prototype (model-backed).

Run:
    streamlit run app/app.py
"""
import base64
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
sys.path.insert(0, str(ROOT / "rules"))
sys.path.insert(0, str(ROOT / "train"))

from predictors import get_model_bundle  # noqa: E402
from rules_engine import SubmissionInputs, score_submission  # noqa: E402

DATA_PATH = ROOT / "data" / "model_scored_dataset.parquet"
MARKET_ADJ_PATH = ROOT / "models" / "market_adj_by_state.json"
LOGO_PATH = ROOT / "assets" / "logo.png"  # drop your company logo here (PNG/SVG, transparent bg)

st.set_page_config(
    page_title="Personal Auto UW & Pricing",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# THEME / COLOR PALETTE
# ============================================================================
NAVY = "#0B2545"
NAVY_LIGHT = "#13315C"
ACCENT = "#2E86FF"
ACCENT_SOFT = "#E8F1FF"
TEAL = "#0EA5A4"
AMBER = "#F5A623"
RED = "#E5484D"
GREEN = "#2FAE6B"
GRAY_BG = "#F5F7FA"
TEXT_MUTED = "#5A6B87"

BAND_COLORS = {
    "Preferred": GREEN,
    "Standard": ACCENT,
    "Non-Standard": AMBER,
    "Decline": RED,
    "Hard Stop": "#8A1C1C",
}

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, 'Segoe UI', sans-serif", color=NAVY, size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[ACCENT, TEAL, AMBER, RED, GREEN, NAVY_LIGHT],
        title=dict(font=dict(size=16, color=NAVY)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#EEF1F6", zeroline=False, linecolor="#D8DEE9"),
        yaxis=dict(gridcolor="#EEF1F6", zeroline=False, linecolor="#D8DEE9"),
        margin=dict(t=50, l=10, r=10, b=10),
    )
)
px.defaults.template = PLOTLY_TEMPLATE

# ============================================================================
# GLOBAL CSS
# ============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    .stApp {{
        background-color: {GRAY_BG};
    }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }}

    /* ---- Top brand bar ---- */
    .brand-bar {{
        display: flex;
        align-items: center;
        gap: 20px;
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
        padding: 16px 28px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 18px rgba(11,37,69,0.18);
    }}
    .brand-logo {{
        width: 72px;
        height: 72px;
        border-radius: 14px;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 34px;
        flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }}
    .brand-logo img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 14px;
        padding: 6px;
    }}
    .brand-text h1 {{
        color: white;
        font-size: 25px;
        font-weight: 800;
        margin: 0;
        line-height: 1.25;
    }}
    .brand-text p {{
        color: #C6D4EA;
        font-size: 13px;
        margin: 4px 0 0 0;
    }}

    /* ---- KPI cards ---- */
    .kpi-card {{
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(11,37,69,0.08);
        border: 1px solid #EAEEF4;
        height: 100%;
    }}
    .kpi-label {{
        font-size: 12px;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 24px;
        font-weight: 800;
        color: {NAVY};
    }}
    .kpi-sub {{
        font-size: 12px;
        color: {TEXT_MUTED};
        margin-top: 2px;
    }}

    /* ---- Section headers ---- */
    .section-title {{
        font-size: 15px;
        font-weight: 700;
        color: {NAVY};
        margin: 6px 0 10px 0;
        padding-left: 10px;
        border-left: 4px solid {ACCENT};
    }}

    /* ---- Chart card wrapper ---- */
    .chart-card {{
        background: white;
        border-radius: 12px;
        padding: 14px 16px 4px 16px;
        border: 1px solid #EAEEF4;
        box-shadow: 0 1px 3px rgba(11,37,69,0.06);
        margin-bottom: 18px;
    }}

    /* ---- Result banner ---- */
    .result-banner {{
        border-radius: 14px;
        padding: 20px 24px;
        color: white;
        margin: 8px 0 18px 0;
    }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: white;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #EAEEF4;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        color: {TEXT_MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ACCENT_SOFT} !important;
        color: {ACCENT} !important;
    }}

    /* ---- Buttons ---- */
    .stButton>button, .stFormSubmitButton>button {{
        background: linear-gradient(135deg, {ACCENT} 0%, {NAVY_LIGHT} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        transition: opacity 0.15s ease;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{
        opacity: 0.88;
        color: white;
    }}

    /* ---- Dataframe ---- */
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #EAEEF4;
    }}

    /* ---- Metric widget cleanup (native st.metric) ---- */
    [data-testid="stMetric"] {{
        background: white;
        border: 1px solid #EAEEF4;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(11,37,69,0.06);
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
        font-weight: 600;
    }}
    [data-testid="stMetricValue"] {{
        color: {NAVY};
    }}
</style>
""", unsafe_allow_html=True)


def _logo_html():
    """Render the company logo if assets/logo.png exists, else a branded fallback glyph."""
    if LOGO_PATH.exists():
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        ext = LOGO_PATH.suffix.lstrip(".") or "png"
        return f'<img src="data:image/{ext};base64,{encoded}" alt="logo"/>'
    return "🚗"  # fallback glyph until a real logo asset is supplied


def kpi_card(label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return get_model_bundle()


@st.cache_data
def load_scored_book():
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def load_market_adj():
    if MARKET_ADJ_PATH.exists():
        return json.loads(MARKET_ADJ_PATH.read_text())
    return {}


bundle = load_models()
book = load_scored_book()
market_adj_lookup = load_market_adj()

# ============================================================================
# BRAND / HEADER BAR  (logo top-left)
# ============================================================================
st.markdown(f"""
<div class="brand-bar">
    <div class="brand-logo">{_logo_html()}</div>
    <div class="brand-text">
        <h1>Personal Auto — Underwriting &amp; Pricing</h1>
        <p>XGBoost frequency &middot; Gamma GLM severity &middot; LightGBM bind propensity &middot; deterministic eligibility/premium rules engine</p>
    </div>
</div>
""", unsafe_allow_html=True)

tab_dashboard, tab_interface = st.tabs(["📊  Dashboard", "🧮  Interface (What-If)"])

# ============================================================================
# DASHBOARD TAB
# ============================================================================
with tab_dashboard:
    section_title("Filters")
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        states = ["All States"] + sorted(book["State"].dropna().unique().tolist())
        f_state = c1.selectbox("State", states)
        bands = ["All Risk Bands"] + sorted(book["uw_risk_band"].dropna().unique().tolist())
        f_band = c2.selectbox("Risk Band", bands)
        channels = ["All Channels"] + sorted(book["Submission_Channel"].dropna().unique().tolist())
        f_channel = c3.selectbox("Channel", channels)
        nr = ["All", "New Business", "Renewal"]
        f_nr = c4.selectbox("New/Renewal", nr)
        body_types = ["All Body Types"] + sorted(book["Body_Type"].dropna().unique().tolist())
        f_body = c5.selectbox("Body Type", body_types)

    filtered = book.copy()
    if f_state != "All States":
        filtered = filtered[filtered["State"] == f_state]
    if f_band != "All Risk Bands":
        filtered = filtered[filtered["uw_risk_band"] == f_band]
    if f_channel != "All Channels":
        filtered = filtered[filtered["Submission_Channel"] == f_channel]
    if f_nr != "All":
        filtered = filtered[filtered["New_Renewal_Label"] == f_nr]
    if f_body != "All Body Types":
        filtered = filtered[filtered["Body_Type"] == f_body]

    section_title("Portfolio Overview")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        kpi_card("Submissions", f"{len(filtered):,}")
    with k2:
        kpi_card("Avg Composite Score", f"{filtered['composite_score'].mean():.1f}" if len(filtered) else "–")
    with k3:
        stp_rate = (filtered["recommended_action"] == "Auto-Quote / STP").mean() if len(filtered) else 0
        kpi_card("STP Rate", f"{stp_rate:.1%}")
    with k4:
        decline_rate = (filtered["uw_risk_band"] == "Decline").mean() if len(filtered) else 0
        kpi_card("Decline Rate", f"{decline_rate:.1%}")
    with k5:
        kpi_card("Avg Quoted Premium", f"${filtered['final_quoted_premium'].mean():,.0f}" if len(filtered) else "–")
    with k6:
        kpi_card("Avg Loss Ratio", f"{filtered['expected_loss_ratio'].mean():.1%}" if len(filtered) else "–",
                  sub="current-rate basis")

    st.write("")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        section_title("Risk Band Distribution")
        band_counts = filtered["uw_risk_band"].value_counts().reset_index()
        band_counts.columns = ["Band", "Count"]
        order = [b for b in ["Preferred", "Standard", "Non-Standard", "Decline", "Hard Stop"] if b in band_counts["Band"].values]
        fig = px.bar(
            band_counts, x="Band", y="Count", color="Band",
            category_orders={"Band": order},
            color_discrete_map=BAND_COLORS,
            text="Count",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(showlegend=False, height=360, xaxis_title=None, yaxis_title="Submissions")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        section_title("Recommended Action Mix")
        action_counts = filtered["recommended_action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        fig2 = px.pie(
            action_counts, names="Action", values="Count", hole=0.55,
            color_discrete_sequence=[ACCENT, TEAL, AMBER, RED, NAVY_LIGHT, GREEN],
        )
        fig2.update_traces(textinfo="percent+label", textfont_size=12, marker=dict(line=dict(color="white", width=2)))
        fig2.update_layout(height=360, legend=dict(orientation="h", yanchor="bottom", y=-0.25))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        section_title("Avg Composite Score by State")
        by_state = filtered.groupby("State")["composite_score"].mean().reset_index().sort_values(
            "composite_score", ascending=False)
        fig3 = px.bar(by_state, x="State", y="composite_score", color="composite_score",
                       color_continuous_scale=[ACCENT_SOFT, ACCENT, NAVY])
        fig3.update_layout(height=380, xaxis_title=None, yaxis_title="Avg Composite Score",
                            coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        section_title("Technical vs. Final Quoted Premium (sample)")
        sample = filtered.sample(min(500, len(filtered)), random_state=1) if len(filtered) else filtered
        fig4 = px.scatter(
            sample, x="technical_premium", y="final_quoted_premium",
            color="uw_risk_band", opacity=0.7,
            color_discrete_map=BAND_COLORS,
            labels={"technical_premium": "Technical Premium ($)",
                    "final_quoted_premium": "Final Quoted Premium ($)"},
        )
        fig4.add_shape(type="line", x0=0, y0=0, x1=16000, y1=16000,
                        line=dict(dash="dash", color=TEXT_MUTED, width=1.5))
        fig4.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=-0.35))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    section_title("Rate Adequacy Exceptions")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    exceptions = filtered[filtered["rate_adequacy_flag"] == True]
    pct = len(exceptions) / max(len(filtered), 1)
    st.markdown(
        f'<p style="color:{TEXT_MUTED}; font-size:13px; margin-top:-4px;">'
        f'<b style="color:{NAVY}">{len(exceptions):,}</b> of {len(filtered):,} submissions '
        f'(<b style="color:{RED if pct > 0.1 else NAVY}">{pct:.1%}</b>) flagged — final quoted premium below technical premium</p>',
        unsafe_allow_html=True,
    )
    show_cols = ["Policy_ID", "State", "uw_risk_band", "technical_premium", "final_quoted_premium",
                 "composite_score", "recommended_action"]
    st.dataframe(exceptions[show_cols].head(50), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# INTERFACE TAB (live what-if simulator)
# ============================================================================
with tab_interface:
    section_title("Live Underwriting & Pricing Simulator")
    st.markdown(
        f'<p style="color:{TEXT_MUTED}; font-size:13px; margin-top:-8px;">Enter a hypothetical submission. '
        f'Frequency, severity, and bind probability come from the trained models; eligibility and pricing '
        f'use the deterministic rules engine.</p>',
        unsafe_allow_html=True,
    )

    with st.form("interface_form"):
        st.markdown("##### 🧍 Driver & Loss Risk Parameters")
        c1, c2, c3, c4, c5 = st.columns(5)
        driver_age = c1.number_input("Driver Age", 16, 99, 30)
        years_licensed = c2.number_input("Years Licensed", 0, 80, 10)
        annual_mileage = c3.number_input("Annual Mileage", 0, 100000, 12000, step=500)
        vehicle_use = c4.selectbox("Vehicle Use", ["Pleasure", "Commute", "Business"])
        territory_type = c5.selectbox("Territory Type", ["Suburban", "Urban", "Rural"])

        c1, c2, c3, c4 = st.columns(4)
        traffic_congestion = c1.slider("Traffic Congestion (0-10)", 0, 10, 5)
        vehicle_theft_risk = c2.slider("Vehicle Theft Risk (0-10)", 0, 10, 5)
        coverage_lapse_days = c3.number_input("Coverage Lapse (days)", 0, 365, 0)
        undeclared_business_use = c4.checkbox("Undeclared Business Use?")

        st.divider()
        st.markdown("##### 📋 Driving & Claims History")
        c1, c2, c3, c4, c5 = st.columns(5)
        prior_claims_3y = c1.number_input("Prior Claims (3Y)", 0, 20, 0)
        at_fault_claims_3y = c2.number_input("At-Fault Claims (3Y)", 0, 20, 0)
        prior_claims_5y = c3.number_input("Prior Claims (5Y)", 0, 20, 0)
        moving_violations_3y = c4.number_input("Moving Violations (3Y)", 0, 20, 0)
        speeding_violations_3y = c5.number_input("Speeding Violations (3Y)", 0, 20, 0)

        c1, c2, c3 = st.columns(3)
        major_violation = c1.checkbox("Major Violation?")
        dui_flag = c2.checkbox("DUI/DWI Flag?")
        license_suspension = c3.checkbox("License Suspension?")

        st.divider()
        st.markdown("##### 📡 Telematics")
        c1, c2 = st.columns(2)
        telematics_enrolled = c1.checkbox("Telematics Enrolled?")
        telematics_safety_score = c2.slider("Telematics Safety Score (0-100)", 0, 100, 75)

        st.divider()
        st.markdown("##### 🚙 Vehicle & Coverage Detail")
        c1, c2, c3, c4, c5 = st.columns(5)
        vehicle_value = c1.number_input("Vehicle Value ($)", 500, 200000, 28000, step=500)
        repairability_score = c2.slider("Repairability Score (0-10)", 0, 10, 5)
        parts_cost_index = c3.number_input("Parts Cost Index (~0.7-1.5)", 0.5, 2.0, 1.0, step=0.05)
        labor_cost_index = c4.number_input("Labor Cost Index (~0.7-1.5)", 0.5, 2.0, 1.0, step=0.05)
        medical_cost_index = c5.number_input("Medical Cost Index (~0.7-1.6)", 0.5, 2.0, 1.0, step=0.05)

        c1, c2, c3, c4, c5 = st.columns(5)
        litigation_environment = c1.slider("Litigation Environment (0-10)", 0, 10, 5)
        luxury_vehicle = c2.checkbox("Luxury Vehicle?")
        performance_vehicle = c3.checkbox("Performance Vehicle?")
        collision_deductible = c4.number_input("Collision Deductible ($)", 0, 5000, 500, step=100)
        comprehensive_deductible = c5.number_input("Comprehensive Deductible ($)", 0, 5000, 500, step=100)

        st.divider()
        st.markdown("##### 🎯 Appetite / Eligibility Parameters")
        c1, c2, c3, c4 = st.columns(4)
        cat_exposure_score = c1.slider("CAT Exposure Score (0-10)", 0, 10, 5)
        state_in_appetite = c2.checkbox("State In Carrier Appetite?", value=True)
        prior_total_loss = c3.checkbox("Prior Total-Loss Claim?")
        confirmed_fraud = c4.checkbox("Confirmed Fraud Indicator?")

        st.divider()
        st.markdown("##### 🤝 Bind / Market Parameters")
        c1, c2, c3, c4, c5 = st.columns(5)
        new_renewal = c1.selectbox("New / Renewal", ["New Business", "Renewal"])
        customer_tenure_years = c2.number_input("Customer Tenure (years)", 0, 50, 0)
        multi_policy = c3.checkbox("Multi-Policy (bundled)?")
        submission_channel = c4.selectbox("Submission Channel", ["Direct", "Agent", "Aggregator"])
        state = c5.selectbox("State (for market adj.)", sorted(book["State"].dropna().unique().tolist()))

        c1, c2, c3, c4 = st.columns(4)
        agent_engagement = c1.slider("Agent Engagement (0-100)", 0, 100, 5)
        digital_engagement = c2.slider("Digital Engagement (0-100)", 0, 100, 50)
        quote_revisions = c3.number_input("Quote Revisions", 0, 20, 1)
        price_sensitivity = c4.selectbox("Price Sensitivity", ["Low", "Medium", "High"])

        c1, c2 = st.columns(2)
        competitor_premium_estimate = c1.number_input("Competitor Premium Estimate ($, optional)", 0, 30000, 0, step=100)
        prior_year_premium = c2.number_input("Prior Year Premium ($, renewals only)", 0, 30000, 900, step=100)

        target_margin_pct = st.select_slider("Target Profit Margin", options=[2, 3, 4, 5, 6, 7], value=5)

        submitted = st.form_submit_button("🧮  Score Submission", type="primary", use_container_width=True)

    if submitted:
        submission = SubmissionInputs(
            driver_age=driver_age, years_licensed=years_licensed, annual_mileage=annual_mileage,
            vehicle_use=vehicle_use, territory_type=territory_type,
            traffic_congestion=traffic_congestion, vehicle_theft_risk=vehicle_theft_risk,
            coverage_lapse_days=coverage_lapse_days, undeclared_business_use=undeclared_business_use,
            prior_claims_3y=prior_claims_3y, at_fault_claims_3y=at_fault_claims_3y,
            prior_claims_5y=prior_claims_5y, moving_violations_3y=moving_violations_3y,
            speeding_violations_3y=speeding_violations_3y, major_violation=major_violation,
            dui_flag=dui_flag, license_suspension=license_suspension,
            telematics_enrolled=telematics_enrolled, telematics_safety_score=telematics_safety_score,
            vehicle_value=vehicle_value, repairability_score=repairability_score,
            parts_cost_index=parts_cost_index, labor_cost_index=labor_cost_index,
            medical_cost_index=medical_cost_index, litigation_environment=litigation_environment,
            luxury_vehicle=luxury_vehicle, performance_vehicle=performance_vehicle,
            collision_deductible=collision_deductible, comprehensive_deductible=comprehensive_deductible,
            cat_exposure_score=cat_exposure_score, state_in_appetite=state_in_appetite,
            prior_total_loss=prior_total_loss, confirmed_fraud=confirmed_fraud,
            new_renewal=new_renewal, customer_tenure_years=customer_tenure_years,
            multi_policy=multi_policy, submission_channel=submission_channel, state=state,
            agent_engagement=agent_engagement, digital_engagement=digital_engagement,
            quote_revisions=quote_revisions, competing_quotes_count=0,
            price_sensitivity=price_sensitivity,
            competitor_premium_estimate=competitor_premium_estimate if competitor_premium_estimate > 0 else None,
            prior_year_premium=prior_year_premium if new_renewal == "Renewal" else None,
            target_profit_margin=target_margin_pct / 100,
        )

        # Build a 1-row dataframe matching model feature schema for prediction
        row_df = pd.DataFrame([{
            "Driver_Age": driver_age, "Years_Licensed": years_licensed, "Annual_Mileage": annual_mileage,
            "Customer_Tenure_Years": customer_tenure_years, "Coverage_Lapse_Days": coverage_lapse_days,
            "Prior_Claims_3Y": prior_claims_3y, "At_Fault_Claims_3Y": at_fault_claims_3y,
            "Prior_Claims_5Y": prior_claims_5y, "At_Fault_Claims_5Y": at_fault_claims_3y,
            "Moving_Violations_3Y": moving_violations_3y, "Speeding_Violations_3Y": speeding_violations_3y,
            "Months_Since_Last_Claim": 999,
            "Hard_Braking_Rate": 0.0, "Rapid_Acceleration_Rate": 0.0, "Speeding_Rate": 0.0,
            "Night_Driving_Percentage": 0.0, "Distracted_Driving_Score": 0.0,
            "Telematics_Safety_Score": telematics_safety_score,
            "Vehicle_Age": 5, "Vehicle_Value": vehicle_value, "Vehicle_Safety_Rating": 5,
            "Parts_Cost_Index": parts_cost_index, "Labor_Cost_Index": labor_cost_index,
            "Repairability_Score": repairability_score,
            "Traffic_Congestion_Score": traffic_congestion, "Vehicle_Theft_Risk_Score": vehicle_theft_risk,
            "Weather_Risk_Score": 5, "Hail_Risk_Score": 5, "Flood_Risk_Score": 5, "Hurricane_Risk_Score": 5,
            "Litigation_Environment_Score": litigation_environment, "Repair_Cost_Index": 1.0,
            "Medical_Cost_Index": medical_cost_index, "Uninsured_Motorist_Risk": 5,
            "CAT_Exposure_Score": cat_exposure_score,
            "Collision_Deductible": collision_deductible, "Comprehensive_Deductible": comprehensive_deductible,
            "PIP_Limit": 10000,
            "New_Renewal_Label": new_renewal, "Multi_Policy_Flag": 1 if multi_policy else 0,
            "State": state, "Urban_Suburban_Rural": territory_type, "Vehicle_Use": vehicle_use,
            "Marital_Status": "Married", "Young_Driver_Flag": 1 if driver_age < 25 else 0,
            "Senior_Driver_Flag": 1 if driver_age >= 65 else 0,
            "Continuous_Coverage_Flag": 1 if coverage_lapse_days == 0 else 0,
            "Major_Violation_Flag": 1 if major_violation else 0, "DUI_Flag": 1 if dui_flag else 0,
            "License_Suspension_Flag": 1 if license_suspension else 0,
            "Prior_Total_Loss_Flag": 1 if prior_total_loss else 0,
            "Undeclared_Business_Use_Flag": 1 if undeclared_business_use else 0,
            "Telematics_Enrolled": 1 if telematics_enrolled else 0,
            "Body_Type": "Sedan", "EV_Flag": 0, "Luxury_Flag": 1 if luxury_vehicle else 0,
            "Performance_Vehicle_Flag": 1 if performance_vehicle else 0,
            "Anti_Theft_Flag": 0, "ADAS_Flag": 0, "No_Fault_State_Flag": 0,
            "Collision_Flag": 1, "Comprehensive_Flag": 1, "PIP_Flag": 0, "UM_UIM_Flag": 1,
            # bind-model-only fields
            "Prior_Year_Premium": prior_year_premium if new_renewal == "Renewal" else 1200,
            "Quote_Completion_Minutes": 5, "Quote_Revisions": quote_revisions,
            "Agent_Engagement_Score": agent_engagement, "Digital_Engagement_Score": digital_engagement,
            "Price_Sensitivity_Band": price_sensitivity,
            "Submission_Channel": submission_channel,
        }])

        # Technical_Premium isn't known yet at this point - the bind model needs
        # Final_Quoted_Premium, so we bootstrap it using a 2-pass approach:
        # pass 1 with a placeholder premium to get freq/severity, compute premium,
        # then re-run bind prediction using the real quoted premium.
        freq_pred = bundle.predict_frequency(row_df).iloc[0]
        sev_pred = bundle.predict_severity(row_df).iloc[0]
        expected_loss_cost = freq_pred * sev_pred
        loss_propensity_score = float((book["Pred_Expected_Loss_Cost"] < expected_loss_cost).mean() * 100)

        market_adj = market_adj_lookup.get(state, 0.0)

        # Pass 1: score with a bind-probability placeholder to get the premium
        prelim = score_submission(
            x=submission, expected_claim_frequency=freq_pred, expected_severity=sev_pred,
            bind_probability=0.5, loss_propensity_score=loss_propensity_score, market_adj=market_adj,
        )
        row_df["Final_Quoted_Premium"] = prelim.final_quoted_premium
        bind_prob = bundle.predict_bind_probability(row_df).iloc[0]

        # Pass 2: final result with the real bind probability
        result = score_submission(
            x=submission, expected_claim_frequency=freq_pred, expected_severity=sev_pred,
            bind_probability=bind_prob, loss_propensity_score=loss_propensity_score, market_adj=market_adj,
        )

        st.divider()
        band_color = BAND_COLORS.get(result.uw_risk_band, NAVY)
        st.markdown(f"""
        <div class="result-banner" style="background: linear-gradient(135deg, {band_color} 0%, {NAVY} 140%);">
            <div style="font-size:13px; opacity:0.85; text-transform:uppercase; letter-spacing:0.05em;">Underwriting Result</div>
            <div style="font-size:26px; font-weight:800; margin-top:4px;">{result.uw_risk_band}</div>
            <div style="font-size:14px; margin-top:6px; opacity:0.95;">Recommended Action: <b>{result.recommended_action}</b></div>
        </div>
        """, unsafe_allow_html=True)

        if result.hard_stop:
            st.error("🛑 Hard stop triggered: " + " | ".join(result.hard_stop_reasons))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Composite UW Score", f"{result.composite_score}/100")
        m2.metric("Loss Propensity Score", f"{result.loss_propensity_score:.1f}")
        m3.metric("Bind Propensity Score", f"{result.bind_propensity_score:.1f}")
        m4.metric("Appetite Score", f"{result.appetite_score:.1f}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Claim Frequency", f"{result.expected_claim_frequency:.3f}")
        m2.metric("Predicted Severity", f"${result.expected_severity:,.0f}")
        m3.metric("Technical Premium", f"${result.technical_premium:,.0f}")
        m4.metric("Final Quoted Premium", f"${result.final_quoted_premium:,.0f}",
                  delta=f"{(result.final_quoted_premium/result.technical_premium - 1):+.1%} vs technical")

        # Quick visual: premium build-up (technical -> final) and score gauges
        st.write("")
        col_x, col_y = st.columns([1.3, 1])
        with col_x:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            section_title("Technical vs. Final Premium")
            comp_df = pd.DataFrame({
                "Type": ["Technical Premium", "Final Quoted Premium"],
                "Amount": [result.technical_premium, result.final_quoted_premium],
            })
            fig5 = px.bar(comp_df, x="Type", y="Amount", color="Type", text="Amount",
                          color_discrete_sequence=[NAVY_LIGHT, ACCENT])
            fig5.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", marker_line_width=0)
            fig5.update_layout(showlegend=False, height=300, xaxis_title=None, yaxis_title="Premium ($)")
            st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_y:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            section_title("Composite Score")
            fig6 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.composite_score,
                number={"suffix": " / 100", "font": {"color": NAVY, "size": 30}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED},
                    "bar": {"color": band_color},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "#FDEDED"},
                        {"range": [40, 70], "color": "#FFF6E5"},
                        {"range": [70, 100], "color": "#E9F9F0"},
                    ],
                },
            ))
            fig6.update_layout(height=300, margin=dict(t=20, b=10, l=20, r=20))
            st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        if result.rate_adequacy_flag:
            st.warning("⚠️ Rate Adequacy Exception: final quoted premium is below technical premium.")

        with st.expander("Full calculation detail"):
            st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.__dict__.items()})