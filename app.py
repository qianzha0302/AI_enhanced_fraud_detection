"""
Streamlit UI — infographic-style dark theme.
4-step visual flow: Payment Request → AI Analysis → Risk Score → Action Taken
Supports E-commerce, Investment Banking, and PayPal fraud scenarios.

Run with:  streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import os, sys, pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import get_demo_cases
from fraud_detection_engine import FraudDetectionPipeline
from config import IF_THRESHOLD, INDUSTRY_LABELS
import database as db

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,158,255,0.25);
    border-radius: 16px;
    padding: 20px 18px;
    box-sizing: border-box;
}
.card-title {
    color: #4a9eff;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.step-badge {
    display: inline-block;
    background: rgba(0,212,255,0.15);
    color: #4a9eff;
    border-radius: 50%;
    width: 26px; height: 26px;
    text-align: center;
    line-height: 26px;
    font-weight: 900; font-size: 0.85rem;
    margin-right: 8px;
    vertical-align: middle;
}
.card-section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e8f0fe;
    margin: 6px 0 14px;
}

.field-row   { margin: 8px 0; }
.field-label { color: #8fa8bc; font-size: 0.70rem; letter-spacing: 1px; text-transform: uppercase; }
.field-value { color: #e8f0fe; font-size: 0.94rem; font-weight: 600; margin-top: 1px; }
.field-value.amount { font-size: 1.5rem; color: #4a9eff; font-weight: 800; }
.badge-ok    { color: #4caf50; }
.badge-warn  { color: #ff9800; }
.badge-bad   { color: #f44336; }

.pay-btn {
    margin-top: 18px;
    background: linear-gradient(90deg, #0066cc, #4a9eff);
    color: white; border: none; border-radius: 8px;
    padding: 10px 0; width: 100%;
    font-weight: 800; font-size: 1rem; letter-spacing: 2px;
    text-align: center;
}

.analysis-item {
    display: flex; align-items: flex-start; gap: 10px;
    margin: 10px 0; padding: 9px 11px;
    background: rgba(0,212,255,0.06);
    border-radius: 8px;
    border-left: 3px solid rgba(0,212,255,0.5);
}
.a-icon  { font-size: 1.1rem; flex-shrink: 0; margin-top: 2px; }
.a-title { color: #4a9eff; font-size: 0.82rem; font-weight: 700; }
.a-desc  { color: #a8bfd4; font-size: 0.74rem; margin-top: 2px; }

.arrow-col {
    display: flex; align-items: flex-start;
    justify-content: center; padding-top: 80px;
    color: #4a9eff; font-size: 1.8rem; opacity: 0.6;
}

.decision-box {
    border-radius: 14px; padding: 32px 16px; text-align: center;
    min-height: 260px; box-sizing: border-box;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.d-pass    { background:linear-gradient(145deg,#0a2d14,#0d3d1a); border:2px solid #4caf50; }
.d-monitor { background:linear-gradient(145deg,#1e2d00,#2a3d06); border:2px solid #cddc39; }
.d-review  { background:linear-gradient(145deg,#2d1800,#3d2200); border:2px solid #ff9800; }
.d-block   { background:linear-gradient(145deg,#2d0a0a,#3d1010); border:2px solid #f44336; }
.d-shield  { font-size: 3.2rem; margin-bottom: 10px; }
.d-label   { font-size: 1.7rem; font-weight: 900; letter-spacing: 3px; margin-bottom: 10px; }
.d-label.pass    { color:#4caf50; }
.d-label.monitor { color:#cddc39; }
.d-label.review  { color:#ff9800; }
.d-label.block   { color:#f44336; }
.d-prob    { font-size: 2rem; font-weight: 800; margin-bottom: 6px; }
.d-desc    { color:#a8bfd4; font-size:0.8rem; line-height:1.5; }

.benefit {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 12px; padding: 14px 10px; text-align: center;
}
.b-icon  { font-size: 1.6rem; margin-bottom: 6px; }
.b-title { color: #4a9eff; font-size: 0.8rem; font-weight: 700; }
.b-desc  { color: #8fa8bc; font-size: 0.72rem; margin-top: 4px; line-height: 1.4; }

.llm-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 10px; padding: 14px 16px;
    font-family: monospace; font-size: 0.84rem;
    white-space: pre-wrap; color: #c8d8e8; line-height: 1.6;
}

.industry-tag {
    display: inline-block;
    background: rgba(74,158,255,0.15);
    color: #4a9eff;
    border: 1px solid rgba(74,158,255,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─── Init ─────────────────────────────────────────────────────────────────────
db_available = db.is_available()
if db_available:
    db.init_db()
    db.seed_training_data_from_json(os.path.dirname(__file__) or ".")
    from data_generator import get_demo_user_profiles
    db.seed_user_profiles(get_demo_user_profiles())

@st.cache_resource
def load_pipeline():
    return FraudDetectionPipeline()

pipeline   = load_pipeline()
ALL_CASES  = get_demo_cases()

# ─── Gauge chart ──────────────────────────────────────────────────────────────
def make_gauge(fraud_prob: float) -> go.Figure:
    score = round(fraud_prob * 100, 1)
    color = "#4caf50" if score < 25 else ("#ff9800" if score < 65 else "#f44336")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix":"/100","font":{"size":30,"color":color},"valueformat":".0f"},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#7a95aa","tickfont":{"color":"#7a95aa","size":10}},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
            "steps":[
                {"range":[0,25],"color":"rgba(76,175,80,0.12)"},
                {"range":[25,65],"color":"rgba(255,152,0,0.12)"},
                {"range":[65,100],"color":"rgba(244,67,54,0.12)"},
            ],
            "threshold":{"line":{"color":color,"width":5},"thickness":0.8,"value":score},
        },
        title={"text":f"<b style='color:{color};font-size:13px'>"
               f"{'LOW' if score<25 else 'MEDIUM' if score<65 else 'HIGH'} RISK</b>"},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(t=50,b=0,l=20,r=20), font={"color":"white"},
    )
    return fig


# ─── Industry-specific card renderers ────────────────────────────────────────

def render_payment_card(txn: dict):
    industry = txn.get("industry","ecommerce")

    if industry == "ecommerce":
        off_hours = txn["transaction_time_hour"] < 7 or txn["transaction_time_hour"] > 21
        new_badge = "🆕 First time" if txn.get("is_new_country") else "✅ Known"
        st.markdown(f"""
        <div class="card">
          <div class="card-title"><span class="step-badge">1</span>Payment Request</div>
          <div class="industry-tag">🛒 E-Commerce Wire</div>
          <div class="field-row"><div class="field-label">Card / Account</div>
            <div class="field-value">•••• •••• •••• 1234</div></div>
          <div class="field-row"><div class="field-label">Amount</div>
            <div class="field-value amount">${txn['amount']:,.0f}</div></div>
          <div class="field-row"><div class="field-label">Merchant</div>
            <div class="field-value">{txn.get('merchant_name','—')}</div></div>
          <div class="field-row"><div class="field-label">Category</div>
            <div class="field-value">{txn.get('merchant_category','—')}</div></div>
          <div class="field-row"><div class="field-label">Destination</div>
            <div class="field-value">{txn.get('destination_country','—')} &nbsp; {new_badge}</div></div>
          <div class="field-row"><div class="field-label">Time</div>
            <div class="field-value">{'⚠️ ' if off_hours else ''}{txn.get('transaction_time_hour',0):02d}:00
            &nbsp; (typical: {txn.get('user_typical_hour',9):02d}:00)</div></div>
          <div class="pay-btn">PAY NOW</div>
        </div>""", unsafe_allow_html=True)

    elif industry == "investment_banking":
        ofac_badge  = '<span class="badge-bad">🚨 HIT</span>'  if txn.get("ofac_hit") else '<span class="badge-ok">✅ CLEAR</span>'
        swift_badge = '<span class="badge-ok">✅ Verified</span>' if txn.get("swift_verified") else '<span class="badge-warn">⚠️ Unverified</span>'
        docs_badge  = '<span class="badge-ok">✅ Attached</span>' if txn.get("trade_doc_present") else '<span class="badge-warn">⚠️ Missing</span>'
        new_badge   = "🆕 First wire" if txn.get("is_new_counterparty") else "✅ Known"
        bec_score   = txn.get("bec_risk_score", 0)
        bec_color   = "badge-ok" if bec_score < 0.3 else ("badge-warn" if bec_score < 0.6 else "badge-bad")
        st.markdown(f"""
        <div class="card">
          <div class="card-title"><span class="step-badge">1</span>Wire Transfer</div>
          <div class="industry-tag">🏦 Investment Banking — SWIFT</div>
          <div class="field-row"><div class="field-label">Originator</div>
            <div class="field-value">{txn.get('user_id','—')}</div></div>
          <div class="field-row"><div class="field-label">Amount</div>
            <div class="field-value amount">${txn['amount']:,.0f}</div></div>
          <div class="field-row"><div class="field-label">Counterparty</div>
            <div class="field-value">{txn.get('counterparty_name','—')} &nbsp; {new_badge}</div></div>
          <div class="field-row"><div class="field-label">Destination Country</div>
            <div class="field-value">{txn.get('counterparty_country','—')}</div></div>
          <div class="field-row"><div class="field-label">SWIFT / BIC</div>
            <div class="field-value">{swift_badge}</div></div>
          <div class="field-row"><div class="field-label">OFAC / Sanctions</div>
            <div class="field-value">{ofac_badge}</div></div>
          <div class="field-row"><div class="field-label">BEC Risk Score</div>
            <div class="field-value"><span class="{bec_color}">{bec_score:.2f}/1.0</span></div></div>
          <div class="field-row"><div class="field-label">Trade Docs</div>
            <div class="field-value">{docs_badge}</div></div>
          <div class="pay-btn">SUBMIT WIRE</div>
        </div>""", unsafe_allow_html=True)

    else:  # payment_platform
        dev_badge  = '<span class="badge-bad">⚠️ New Device</span>'  if txn.get("new_device") else '<span class="badge-ok">✅ Trusted</span>'
        ip_badge   = '<span class="badge-warn">⚠️ Mismatch</span>' if txn.get("ip_country_mismatch") else '<span class="badge-ok">✅ Match</span>'
        rec_badge  = "🆕 New" if txn.get("new_recipient") else "✅ Known"
        vel        = txn.get("transactions_last_hour", 1)
        vel_color  = "badge-ok" if vel <= 5 else ("badge-warn" if vel <= 15 else "badge-bad")
        fails      = txn.get("failed_logins_24h", 0)
        fail_color = "badge-ok" if fails == 0 else ("badge-warn" if fails <= 3 else "badge-bad")
        st.markdown(f"""
        <div class="card">
          <div class="card-title"><span class="step-badge">1</span>Payment Request</div>
          <div class="industry-tag">💳 PayPal — P2P / Checkout</div>
          <div class="field-row"><div class="field-label">Account</div>
            <div class="field-value">{txn.get('user_id','—')}</div></div>
          <div class="field-row"><div class="field-label">Amount</div>
            <div class="field-value amount">${txn['amount']:,.0f}</div></div>
          <div class="field-row"><div class="field-label">Recipient</div>
            <div class="field-value">{txn.get('recipient_name','—')} &nbsp; {rec_badge}</div></div>
          <div class="field-row"><div class="field-label">Device</div>
            <div class="field-value">{dev_badge} &nbsp; trust: {txn.get('device_trust_score',1):.2f}</div></div>
          <div class="field-row"><div class="field-label">IP Country</div>
            <div class="field-value">{txn.get('ip_country','US')} &nbsp; {ip_badge}</div></div>
          <div class="field-row"><div class="field-label">Velocity (last hour)</div>
            <div class="field-value"><span class="{vel_color}">{vel} txns</span></div></div>
          <div class="field-row"><div class="field-label">Failed Logins (24h)</div>
            <div class="field-value"><span class="{fail_color}">{fails}</span></div></div>
          <div class="pay-btn">SEND MONEY</div>
        </div>""", unsafe_allow_html=True)


def render_analysis_card(txn: dict, result: dict | None):
    industry   = txn.get("industry","ecommerce")
    if_score   = result["if_score"]   if result else 0.0
    fraud_prob = result["fraud_probability"] if result else 0.0
    is_anom    = if_score < IF_THRESHOLD     if result else False

    if industry == "ecommerce":
        items = [
            ("👤","Behavior Analysis",
             f"Acct age {txn.get('account_age_days','—')}d · "
             f"Dispute {txn.get('user_dispute_rate',0)}% · "
             f"{txn.get('user_transaction_frequency',0)} txns/mo"),
            ("📍","Geography & Country",
             f"{'🆕 New country' if txn.get('is_new_country') else '✅ Known corridor'} · "
             f"Intl history: {txn.get('user_international_frequency',0)} wires"),
            ("🔀","Isolation Forest",
             f"Anomaly score: {if_score:.3f} → "
             f"{'🟠 Anomalous' if is_anom else '🟢 Normal'} (threshold {IF_THRESHOLD})"),
            ("📊","XGBoost Score",
             f"Amount deviation ×{txn.get('amount_deviation',1):.1f} · "
             f"Fraud probability: {fraud_prob:.1%}"),
        ]
    elif industry == "investment_banking":
        items = [
            ("🏛️","Counterparty Risk",
             f"Country risk: {txn.get('counterparty_risk_score',0):.2f}/1.0 · "
             f"OFAC: {'🚨 HIT' if txn.get('ofac_hit') else '✅ Clear'} · "
             f"SWIFT: {'✅' if txn.get('swift_verified') else '⚠️ Unverified'}"),
            ("📧","BEC / Email Fraud",
             f"BEC risk score: {txn.get('bec_risk_score',0):.2f}/1.0 · "
             f"{'🆕 First wire' if txn.get('is_new_counterparty') else '✅ Known counterparty'} · "
             f"Trade docs: {'✅' if txn.get('trade_doc_present') else '⚠️ Missing'}"),
            ("🔀","Isolation Forest",
             f"Anomaly score: {if_score:.3f} → "
             f"{'🟠 Anomalous' if is_anom else '🟢 Normal'} (threshold {IF_THRESHOLD})"),
            ("📊","XGBoost Score",
             f"Amount ×{txn.get('amount_vs_monthly_avg',1):.1f}x monthly avg · "
             f"Fraud probability: {fraud_prob:.1%}"),
        ]
    else:  # payment_platform
        items = [
            ("🔐","Account Takeover Signals",
             f"Failed logins: {txn.get('failed_logins_24h',0)} · "
             f"New device: {'Yes ⚠️' if txn.get('new_device') else 'No ✅'} · "
             f"IP mismatch: {'Yes ⚠️' if txn.get('ip_country_mismatch') else 'No ✅'}"),
            ("⚡","Velocity Analysis",
             f"{txn.get('transactions_last_hour',1)} txns in last hour · "
             f"Device trust: {txn.get('device_trust_score',1):.2f}/1.0 · "
             f"{'🆕 New recipient' if txn.get('new_recipient') else '✅ Known recipient'}"),
            ("🔀","Isolation Forest",
             f"Anomaly score: {if_score:.3f} → "
             f"{'🟠 Anomalous' if is_anom else '🟢 Normal'} (threshold {IF_THRESHOLD})"),
            ("📊","XGBoost Score",
             f"Amount ×{txn.get('amount_deviation',1):.1f}x typical · "
             f"Fraud probability: {fraud_prob:.1%}"),
        ]

    items_html = "".join(
        f'<div class="analysis-item"><div class="a-icon">{icon}</div>'
        f'<div><div class="a-title">{title}</div>'
        f'<div class="a-desc">{desc}</div></div></div>'
        for icon, title, desc in items
    )
    st.markdown(f"""
    <div class="card">
        <div class="card-title"><span class="step-badge">2</span>AI Analysis in Real Time</div>
        <div class="card-section-title" style="margin-bottom:10px">Processing...</div>
        {items_html}
    </div>""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    st.caption("GenAI-Enhanced 4-Stage Pipeline")
    st.divider()

    # Industry selector
    industry_key = st.radio(
        "Industry vertical",
        options=["ecommerce","investment_banking","payment_platform"],
        format_func=lambda x: INDUSTRY_LABELS[x],
        index=0,
    )
    st.divider()

    mode = st.radio("Input mode", ["Demo Cases", "Custom Transaction"], index=0)

    cases = ALL_CASES[industry_key]

    if mode == "Demo Cases":
        selected_id = st.selectbox(
            "Select demo case",
            options=[c["transaction_id"] for c in cases],
            format_func=lambda x: next(
                (f"{c['transaction_id']} — {c.get('_label','')}" for c in cases if c["transaction_id"]==x), x),
        )
        txn = next(c for c in cases if c["transaction_id"] == selected_id)

    else:
        # Custom transaction — fields differ by industry
        if industry_key == "ecommerce":
            st.subheader("E-Commerce Transaction")
            amount      = st.number_input("Amount ($)", min_value=1, value=10000, step=500)
            user_avg    = st.number_input("User avg amount ($)", min_value=1, value=3000, step=500)
            user_std    = st.number_input("User std ($)", min_value=1, value=800, step=100)
            destination = st.text_input("Destination country", value="Nigeria")
            merchant    = st.text_input("Merchant name", value="Lagos Export Ltd")
            merch_cat   = st.text_input("Merchant category", value="International Trade")
            is_new      = st.checkbox("First transaction to this country", value=True)
            hour        = st.slider("Transaction hour", 0, 23, 3)
            account_age = st.number_input("Account age (days)", min_value=1, value=730, step=30)
            dispute_rate= st.slider("Dispute rate (%)", 0.0, 10.0, 0.0, step=0.1)
            tx_freq     = st.slider("Transactions/month", 1, 100, 25)
            intl_freq   = st.slider("Prior international wires", 0, 50, 0)
            txn = {
                "transaction_id":"CUSTOM_001","user_id":"USR_CUSTOM",
                "industry":"ecommerce",
                "amount":float(amount),"user_avg_amount":float(user_avg),
                "user_std":float(user_std),"amount_deviation":round(amount/user_avg,2),
                "merchant_name":merchant,"merchant_category":merch_cat,
                "destination_country":destination,"is_new_country":is_new,
                "transaction_time_hour":hour,"payment_method":"Wire",
                "user_international_frequency":intl_freq,
                "account_age_days":account_age,"user_industry":"Retail",
                "user_primary_location":"Chicago, IL","user_dispute_rate":dispute_rate,
                "user_transaction_frequency":tx_freq,"user_typical_hour":11,
                "user_fraud_history":0,"label":-1,
            }

        elif industry_key == "investment_banking":
            st.subheader("Wire Transfer")
            amount       = st.number_input("Amount ($)", min_value=10000, value=2_000_000, step=100000)
            monthly_vol  = st.number_input("Monthly wire volume ($)", min_value=100000, value=10_000_000, step=500000)
            cp_name      = st.text_input("Counterparty name", value="Global Parts Supply Ltd")
            cp_country   = st.text_input("Counterparty country", value="Cayman Islands")
            cp_risk      = st.slider("Counterparty risk score", 0.0, 1.0, 0.7, step=0.05)
            swift_ok     = st.checkbox("SWIFT BIC verified", value=True)
            ofac_hit     = st.checkbox("OFAC / Sanctions HIT", value=False)
            bec_risk     = st.slider("BEC risk score", 0.0, 1.0, 0.8, step=0.05)
            new_cp       = st.checkbox("First wire to this counterparty", value=True)
            trade_doc    = st.checkbox("Trade documents attached", value=False)
            hour         = st.slider("Initiated at (hour)", 0, 23, 22)
            account_age  = st.number_input("Client account age (days)", min_value=1, value=2000, step=30)
            txn = {
                "transaction_id":"CUSTOM_IB_001","user_id":"INST_CUSTOM",
                "industry":"investment_banking",
                "amount":float(amount),"monthly_wire_volume":float(monthly_vol),
                "amount_vs_monthly_avg":round(amount/(monthly_vol/12),2),
                "counterparty_name":cp_name,"counterparty_bank":cp_name,
                "counterparty_country":cp_country,"counterparty_iban":"CUSTOM",
                "counterparty_risk_score":cp_risk,
                "swift_verified":int(swift_ok),"bec_risk_score":bec_risk,
                "ofac_hit":int(ofac_hit),"is_new_counterparty":int(new_cp),
                "trade_doc_present":int(trade_doc),
                "transaction_time_hour":hour,"account_age_days":account_age,
                "user_dispute_rate":0.2,"client_industry":"Finance",
                "aml_risk_tier":"Medium","relationship_manager":"RM_CUSTOM",
                "payment_purpose":"Business Transaction","label":-1,
            }

        else:  # payment_platform
            st.subheader("PayPal Payment")
            amount      = st.number_input("Amount ($)", min_value=1, value=300, step=10)
            user_avg    = st.number_input("User avg amount ($)", min_value=1, value=80, step=10)
            recipient   = st.text_input("Recipient name", value="Unknown_8847")
            rec_age     = st.number_input("Recipient account age (days)", min_value=0, value=5, step=1)
            new_rec     = st.checkbox("New recipient", value=True)
            new_dev     = st.checkbox("New / unrecognised device", value=True)
            dev_trust   = st.slider("Device trust score", 0.0, 1.0, 0.1, step=0.05)
            ip_country  = st.text_input("IP country", value="RU")
            ip_mm       = st.checkbox("IP country mismatch", value=True)
            failed_log  = st.slider("Failed logins (24h)", 0, 20, 8)
            velocity    = st.slider("Transactions last hour", 1, 50, 3)
            account_age = st.number_input("Account age (days)", min_value=1, value=900, step=30)
            txn = {
                "transaction_id":"CUSTOM_PP_001","user_id":"PP_CUSTOM",
                "industry":"payment_platform",
                "amount":float(amount),"user_avg_amount":float(user_avg),
                "user_std":float(user_avg*0.3),"amount_deviation":round(amount/user_avg,2),
                "recipient_name":recipient,"recipient_account_age_days":int(rec_age),
                "new_recipient":int(new_rec),"new_device":int(new_dev),
                "device_trust_score":dev_trust,"ip_country":ip_country,
                "ip_country_mismatch":int(ip_mm),"failed_logins_24h":failed_log,
                "transactions_last_hour":velocity,"account_age_days":account_age,
                "user_dispute_rate":0.5,"account_verified":True,"user_primary_country":"US",
                "label":-1,
            }

    st.divider()
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    st.divider()
    llm_ok = bool(os.environ.get("SILICONFLOW_API_KEY"))
    if llm_ok:
        st.success("LLM: DeepSeek-V3 ✓")
    else:
        st.warning("LLM: Mock mode")
    if db_available:
        st.success("DB: Neon connected ✓")
    else:
        st.warning("DB: Not connected")


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#e8f0fe;font-size:2rem;margin-bottom:2px'>"
    "How AI is Improving Fraud Detection</h1>"
    "<p style='text-align:center;color:#4a9eff;font-size:1.1rem;margin-bottom:4px'>"
    "E-Commerce · Investment Banking · Payment Platforms</p>"
    "<p style='text-align:center;color:#8fa8bc;font-size:0.85rem;margin-bottom:20px'>"
    "AI analyzes data in real time, detects suspicious patterns, "
    "and helps prevent fraud before it happens.</p>",
    unsafe_allow_html=True,
)

tab_main, tab_history, tab_docs = st.tabs(["🔍 Analysis", "📜 History", "📖 How It Works"])

# ═══════════════════════════════════════════════════════════════
# Tab 1: 4-Step Flow
# ═══════════════════════════════════════════════════════════════
with tab_main:
    if run_btn:
        with st.spinner("AI analyzing transaction..."):
            result = pipeline.process_transaction(txn)
        if db_available:
            try:
                db.save_result(txn, result)
            except Exception as e:
                st.warning(f"DB write failed: {e}")
        st.session_state["last_result"] = result
        st.session_state["last_txn"]    = txn

    result  = st.session_state.get("last_result")
    cur_txn = st.session_state.get("last_txn", txn)

    # If industry switched, clear old result so we don't show mismatched data
    if result and result.get("industry") != cur_txn.get("industry"):
        result  = None
        cur_txn = txn

    c1, arr1, c2, arr2, c3, arr3, c4 = st.columns([3, 0.4, 3.2, 0.4, 2.8, 0.4, 3])

    with c1:
        render_payment_card(cur_txn)

    with arr1:
        st.markdown('<div class="arrow-col">→</div>', unsafe_allow_html=True)

    with c2:
        render_analysis_card(cur_txn, result)

    with arr2:
        st.markdown('<div class="arrow-col">→</div>', unsafe_allow_html=True)

    with c3:
        fp = result["fraud_probability"] if result else 0.0
        st.markdown("""
        <div class="card">
            <div class="card-title"><span class="step-badge">3</span>Risk Decision</div>
            <div class="card-section-title">Risk Score</div>
            <div style="height:8px"></div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(fp), use_container_width=True, config={"displayModeBar":False})
        if result:
            risk_color = "#4caf50" if fp < 0.25 else ("#ff9800" if fp < 0.65 else "#f44336")
            risk_text  = "LOW RISK" if fp < 0.25 else ("MEDIUM RISK" if fp < 0.65 else "HIGH RISK")
            st.markdown(
                f"<p style='text-align:center;color:{risk_color};font-size:0.95rem;"
                f"font-weight:700;margin-top:-8px'>{risk_text}</p>",
                unsafe_allow_html=True,
            )

    with arr3:
        st.markdown('<div class="arrow-col">→</div>', unsafe_allow_html=True)

    with c4:
        if result:
            d = result["decision"]
            d_cfg = {
                "PASS":    ("✅","d-pass",   "pass",   "APPROVED","Payment is successful\nand secure."),
                "MONITOR": ("🟡","d-monitor","monitor","MONITOR","Flagged for\nsoft monitoring."),
                "REVIEW":  ("⚠️","d-review", "review", "REVIEW",  "Manual review required\nbefore processing."),
                "BLOCK":   ("🚨","d-block",  "block",  "BLOCKED", "Suspicious transaction\nblocked."),
            }
            icon, box_cls, lbl_cls, label, desc = d_cfg[d]
            prob_color = {"PASS":"#4caf50","MONITOR":"#cddc39","REVIEW":"#ff9800","BLOCK":"#f44336"}[d]
            st.markdown(f"""
            <div class="decision-box {box_cls}">
                <div class="d-shield">{icon}</div>
                <div class="d-label {lbl_cls}">{label}</div>
                <div class="d-prob" style="color:{prob_color}">{fp:.0%}</div>
                <div class="d-desc">{desc}</div>
                <div style="margin-top:14px;color:#7a95aa;font-size:0.72rem">
                    Stage {result['stage']} · {result['processing_time_ms']:.0f} ms
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="decision-box" style="background:rgba(255,255,255,0.03);
                 border:2px dashed rgba(0,212,255,0.2)">
                <div style="font-size:3rem;margin-bottom:12px">🛡️</div>
                <div style="color:#7a95aa;font-size:0.9rem">Run analysis<br>to see decision</div>
            </div>""", unsafe_allow_html=True)

    # ── LLM Details ───────────────────────────────────────────────────────────
    if result and result.get("llm_analysis"):
        st.write("")
        src = "DeepSeek-V3 (real)" if not pipeline.use_mock else "Mock LLM"
        with st.expander(f"🤖 LLM Context Analysis — Stage 3 & 4  ({src})", expanded=False):
            lc, rc = st.columns(2)
            with lc:
                st.markdown("**Stage 3 — Context Analysis**")
                st.markdown(
                    f'<div class="llm-box">{result["llm_analysis"]}</div>',
                    unsafe_allow_html=True,
                )
            with rc:
                st.markdown("**Stage 4 — Explanations**")
                if result.get("customer_message"):
                    t1, t2 = st.tabs(["📧 Customer / Client Email", "📋 Ops Instructions"])
                    with t1:
                        st.code(result["customer_message"], language=None)
                    with t2:
                        st.code(result.get("ops_instructions",""), language=None)

    # ── Key Benefits bar ──────────────────────────────────────────────────────
    st.write("")
    st.markdown(
        "<p style='text-align:center;color:#4a9eff;font-weight:700;"
        "letter-spacing:2px;font-size:0.85rem;margin-bottom:14px'>KEY BENEFITS</p>",
        unsafe_allow_html=True,
    )
    b1, b2, b3, b4, b5 = st.columns(5)
    benefits = [
        ("🛡️","Real-time Detection","Identifies fraud instantly\nand prevents losses"),
        ("📈","Higher Accuracy","ML models adapt\nto emerging patterns"),
        ("🏦","Multi-Industry","E-Commerce, IB, PayPal\nwith tailored risk signals"),
        ("💰","Lower Chargebacks","Stops fraud before\nit impacts revenue"),
        ("⚡","Scalable & Smart","Handles millions of txns\nwith improving accuracy"),
    ]
    for col, (icon, title, desc) in zip([b1,b2,b3,b4,b5], benefits):
        with col:
            st.markdown(f"""
            <div class="benefit">
                <div class="b-icon">{icon}</div>
                <div class="b-title">{title}</div>
                <div class="b-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;color:#7a95aa;font-size:0.75rem;margin-top:16px'>"
        "🔒 AI-powered fraud detection — E-Commerce · Investment Banking · Payment Platforms</p>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# Tab 2: History
# ═══════════════════════════════════════════════════════════════
with tab_history:
    if not db_available:
        st.warning("Database not connected. Set DATABASE_URL in your .env file.")
    else:
        try:
            stats = db.fetch_stats()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total",       stats["total"])
            c2.metric("✅ Pass",     stats["passed"])
            c3.metric("🟡 Monitor",  stats["monitored"])
            c4.metric("⚠️ Review",   stats["reviewed"])
            c5.metric("🚨 Block",    stats["blocked"])
            st.divider()
        except Exception as e:
            st.error(f"Stats error: {e}")

        try:
            rows = db.fetch_history(50)
            if not rows:
                st.info("No transactions analyzed yet. Run some cases in the Analysis tab.")
            else:
                df = pd.DataFrame(rows)
                df["created_at"]       = pd.to_datetime(df["created_at"]).dt.strftime("%m-%d %H:%M")
                df["fraud_probability"]= (df["fraud_probability"]*100).round(1).astype(str)+"%"
                df["amount"]           = df["amount"].apply(lambda x: f"${float(x):,.0f}")
                df = df.rename(columns={
                    "created_at":"Time","transaction_id":"TXN ID",
                    "user_id":"User","industry":"Industry",
                    "amount":"Amount","amount_deviation":"Dev ×",
                    "fraud_probability":"Fraud Prob","decision":"Decision",
                    "stage":"Stage","processing_time_ms":"ms",
                })
                cols = [c for c in ["Time","TXN ID","User","Industry","Amount",
                                    "Dev ×","Fraud Prob","Decision","Stage","ms"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"History error: {e}")

# ═══════════════════════════════════════════════════════════════
# Tab 3: How It Works
# ═══════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("""
<style>
.doc-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,158,255,0.20);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.doc-h2 {
    color: #4a9eff;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 12px;
}
.doc-rule {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 12px;
    border-radius: 8px;
    margin: 6px 0;
    background: rgba(0,0,0,0.2);
}
.doc-badge {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 20px;
    white-space: nowrap;
    flex-shrink: 0;
    margin-top: 2px;
}
.badge-pass    { background: rgba(76,175,80,0.2);  color: #4caf50; border: 1px solid #4caf50; }
.badge-monitor { background: rgba(205,220,57,0.2); color: #cddc39; border: 1px solid #cddc39; }
.badge-review  { background: rgba(255,152,0,0.2);  color: #ff9800; border: 1px solid #ff9800; }
.badge-block   { background: rgba(244,67,54,0.2);  color: #f44336; border: 1px solid #f44336; }
.doc-rule-text { color: #c8d8e8; font-size: 0.88rem; line-height: 1.6; }
.doc-rule-text b { color: #e8f0fe; }
.feat-chip {
    display: inline-block;
    background: rgba(74,158,255,0.12);
    border: 1px solid rgba(74,158,255,0.3);
    color: #a8ceff;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    margin: 2px 3px;
    font-family: monospace;
}
.stage-arrow {
    text-align: center;
    color: #4a9eff;
    font-size: 1.5rem;
    margin: 4px 0;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)

    st.markdown("<h2 style='color:#e8f0fe;margin-bottom:4px'>System Overview</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8fa8bc;margin-bottom:20px'>This demo shows a <b style='color:#4a9eff'>4-stage AI fraud detection pipeline</b> that routes each transaction through progressively smarter (and more expensive) analysis — only escalating to LLM when the ML models are uncertain.</p>", unsafe_allow_html=True)

    # ── Pipeline flow ─────────────────────────────────────────────
    st.markdown("### 🔄 Pipeline Flow")
    p1, p2, p3, p4, p5, p6, p7 = st.columns([2.5, 0.3, 2.5, 0.3, 2.5, 0.3, 2.5])

    with p1:
        st.markdown("""
        <div class="doc-card" style="border-color:rgba(100,200,255,0.4)">
        <div class="doc-h2">① Isolation Forest</div>
        <div style="color:#8fa8bc;font-size:0.83rem;line-height:1.6">
        <b style="color:#e8f0fe">What it does:</b> Unsupervised anomaly detection — finds statistical outliers in the feature space without needing labels.<br><br>
        <b style="color:#e8f0fe">Algorithm:</b> Isolation Forest (sklearn), 200 trees, contamination=15%<br><br>
        <b style="color:#e8f0fe">Output:</b> Anomaly score (lower = more anomalous)<br><br>
        <b style="color:#e8f0fe">Used for:</b> Surfacing the anomaly signal in the Analysis card. All transactions proceed to Stage 2.
        </div>
        </div>""", unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="stage-arrow" style="padding-top:60px">→</div>', unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="doc-card" style="border-color:rgba(100,200,255,0.4)">
        <div class="doc-h2">② XGBoost Classifier</div>
        <div style="color:#8fa8bc;font-size:0.83rem;line-height:1.6">
        <b style="color:#e8f0fe">What it does:</b> Supervised binary classifier trained on labeled fraud/normal transactions. Each industry has its own model.<br><br>
        <b style="color:#e8f0fe">Algorithm:</b> XGBClassifier, 200 trees, depth=4, class-imbalance weighted<br><br>
        <b style="color:#e8f0fe">Output:</b> Fraud probability 0–100%<br><br>
        <b style="color:#e8f0fe">Decision gates:</b><br>
        &lt; 8% → instant <b style="color:#4caf50">PASS</b><br>
        &gt; 98.5% → instant <b style="color:#f44336">BLOCK</b><br>
        8–98.5% → escalate to LLM
        </div>
        </div>""", unsafe_allow_html=True)

    with p5:
        st.markdown("""
        <div class="doc-card" style="border-color:rgba(100,200,255,0.4)">
        <div class="doc-h2">③ LLM Context Analysis</div>
        <div style="color:#8fa8bc;font-size:0.83rem;line-height:1.6">
        <b style="color:#e8f0fe">What it does:</b> DeepSeek-V3 reads the full transaction + user profile and identifies contextual risk signals that the ML models can't capture.<br><br>
        <b style="color:#e8f0fe">Output:</b> Structured flag analysis:<br>
        🔴 Red Flags (fraud signals)<br>
        🟢 Green Flags (legitimacy signals)<br>
        🟡 Items needing verification<br>
        【Assessment】risk judgment
        </div>
        </div>""", unsafe_allow_html=True)

    with p7:
        st.markdown("""
        <div class="doc-card" style="border-color:rgba(100,200,255,0.4)">
        <div class="doc-h2">④ LLM Explanation</div>
        <div style="color:#8fa8bc;font-size:0.83rem;line-height:1.6">
        <b style="color:#e8f0fe">What it does:</b> Generates human-readable outputs so ops teams can act immediately and customers understand what's happening.<br><br>
        <b style="color:#e8f0fe">Output A — Customer email:</b> Friendly, non-accusatory message explaining the hold and what to do next.<br><br>
        <b style="color:#e8f0fe">Output B — Ops checklist:</b> Step-by-step action items with deadlines and escalation paths.
        </div>
        </div>""", unsafe_allow_html=True)

    with p4:
        st.markdown('<div class="stage-arrow" style="padding-top:60px">→</div>', unsafe_allow_html=True)
    with p6:
        st.markdown('<div class="stage-arrow" style="padding-top:60px">→</div>', unsafe_allow_html=True)

    st.divider()

    # ── Decision rules ────────────────────────────────────────────
    st.markdown("### ⚖️ Decision Rules")
    st.markdown("<p style='color:#8fa8bc;font-size:0.85rem;margin-bottom:14px'>Rules apply after XGBoost scores the transaction. Investment Banking also has a pre-ML OFAC rule.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="doc-card">
    <div class="doc-rule">
        <span class="doc-badge badge-pass">PASS</span>
        <div class="doc-rule-text">
        <b>Fraud probability &lt; 8%</b> — Transaction cleared at Stage 2. No LLM call. Low cost, instant response.<br>
        <span style="color:#7a95aa;font-size:0.8rem">Example: Regular domestic purchase, known merchant, amount within user's normal range.</span>
        </div>
    </div>
    <div class="doc-rule">
        <span class="doc-badge badge-monitor">MONITOR</span>
        <div class="doc-rule-text">
        <b>8% ≤ Fraud probability &lt; 50%</b> — Transaction proceeds, LLM adds context, flagged for passive monitoring.<br>
        <span style="color:#7a95aa;font-size:0.8rem">Example: Known counterparty with a new IBAN, or known recipient but an untrusted device.</span>
        </div>
    </div>
    <div class="doc-rule">
        <span class="doc-badge badge-review">REVIEW</span>
        <div class="doc-rule-text">
        <b>50% ≤ Fraud probability &lt; 98.5%</b> — Transaction held. LLM produces customer email + ops checklist. Manual review required before funds move.<br>
        <span style="color:#7a95aa;font-size:0.8rem">Example: BEC-style changed payment instructions, IP country mismatch with new recipient.</span>
        </div>
    </div>
    <div class="doc-rule">
        <span class="doc-badge badge-block">BLOCK</span>
        <div class="doc-rule-text">
        <b>Fraud probability ≥ 98.5%</b> — Auto-blocked at Stage 2, no LLM cost.<br>
        <b>Investment Banking special rule:</b> OFAC/Sanctions match → instant block regardless of ML score.<br>
        <span style="color:#7a95aa;font-size:0.8rem">Example: 26× amount deviation to high-risk country; OFAC-listed entity; compromised account with 9 failed logins.</span>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Industry features ─────────────────────────────────────────
    st.markdown("### 🏭 Industry-Specific Feature Systems")
    st.markdown("<p style='color:#8fa8bc;font-size:0.85rem;margin-bottom:14px'>Each industry has its own ML model trained on domain-appropriate features. The same transaction fields that matter for a retail wire transfer are irrelevant for a P2P payment.</p>", unsafe_allow_html=True)

    fi1, fi2, fi3 = st.columns(3)

    with fi1:
        st.markdown("""
        <div class="doc-card">
        <div class="doc-h2">🛒 E-Commerce</div>
        <div style="color:#8fa8bc;font-size:0.82rem;margin-bottom:10px">Key risk signals: amount spike vs. user average, first-time destination country, off-hours activity.</div>
        <span class="feat-chip">amount</span>
        <span class="feat-chip">amount_deviation</span>
        <span class="feat-chip">is_new_country</span>
        <span class="feat-chip">transaction_time_hour</span>
        <span class="feat-chip">user_avg_amount</span>
        <span class="feat-chip">user_international_frequency</span>
        <span class="feat-chip">user_dispute_rate</span>
        <span class="feat-chip">user_fraud_history</span>
        <span class="feat-chip">account_age_days</span>
        <span class="feat-chip">user_transaction_frequency</span>
        <span class="feat-chip">user_std</span>
        <div style="margin-top:12px;color:#7a95aa;font-size:0.78rem">
        📊 Top feature importances: <b style="color:#a8ceff">amount (44%)</b>, amount_deviation (26%), is_new_country (21%)
        </div>
        </div>""", unsafe_allow_html=True)

    with fi2:
        st.markdown("""
        <div class="doc-card">
        <div class="doc-h2">🏦 Investment Banking</div>
        <div style="color:#8fa8bc;font-size:0.82rem;margin-bottom:10px">Key risk signals: BEC email fraud score, sanctions screening, counterparty jurisdiction risk, new vs. known beneficiary.</div>
        <span class="feat-chip">bec_risk_score</span>
        <span class="feat-chip">counterparty_risk_score</span>
        <span class="feat-chip">is_new_counterparty</span>
        <span class="feat-chip">ofac_hit</span>
        <span class="feat-chip">amount_vs_monthly_avg</span>
        <span class="feat-chip">swift_verified</span>
        <span class="feat-chip">trade_doc_present</span>
        <span class="feat-chip">transaction_time_hour</span>
        <span class="feat-chip">account_age_days</span>
        <span class="feat-chip">user_dispute_rate</span>
        <span class="feat-chip">amount</span>
        <div style="margin-top:12px;color:#7a95aa;font-size:0.78rem">
        📊 Top feature importances: <b style="color:#a8ceff">bec_risk_score (35%)</b>, counterparty_risk_score (22%), is_new_counterparty (20%), ofac_hit (18%)
        </div>
        </div>""", unsafe_allow_html=True)

    with fi3:
        st.markdown("""
        <div class="doc-card">
        <div class="doc-h2">💳 Payment Platform (PayPal)</div>
        <div style="color:#8fa8bc;font-size:0.82rem;margin-bottom:10px">Key risk signals: device trust score, recipient account age, new vs. known recipient — strong ATO and money mule indicators.</div>
        <span class="feat-chip">device_trust_score</span>
        <span class="feat-chip">recipient_account_age_days</span>
        <span class="feat-chip">new_recipient</span>
        <span class="feat-chip">new_device</span>
        <span class="feat-chip">ip_country_mismatch</span>
        <span class="feat-chip">failed_logins_24h</span>
        <span class="feat-chip">transactions_last_hour</span>
        <span class="feat-chip">amount_deviation</span>
        <span class="feat-chip">account_age_days</span>
        <span class="feat-chip">user_dispute_rate</span>
        <span class="feat-chip">amount</span>
        <div style="margin-top:12px;color:#7a95aa;font-size:0.78rem">
        📊 Top feature importances: <b style="color:#a8ceff">device_trust_score (47%)</b>, recipient_account_age_days (28%), new_recipient (26%)
        </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Demo cases table ──────────────────────────────────────────
    st.markdown("### 🧪 Demo Cases at a Glance")

    st.markdown("""
    <div class="doc-card">
    <table style="width:100%;border-collapse:collapse;font-size:0.83rem">
    <thead>
    <tr style="border-bottom:1px solid rgba(74,158,255,0.3)">
      <th style="color:#4a9eff;padding:6px 10px;text-align:left">Case</th>
      <th style="color:#4a9eff;padding:6px 10px;text-align:left">Industry</th>
      <th style="color:#4a9eff;padding:6px 10px;text-align:left">Scenario</th>
      <th style="color:#4a9eff;padding:6px 10px;text-align:left">Why it's interesting</th>
      <th style="color:#4a9eff;padding:6px 10px;text-align:center">Decision</th>
      <th style="color:#4a9eff;padding:6px 10px;text-align:center">LLM?</th>
    </tr>
    </thead>
    <tbody>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">EC_001</td><td style="color:#8fa8bc;padding:6px 10px">E-commerce</td>
      <td style="color:#e8f0fe;padding:6px 10px">Normal Amazon purchase</td>
      <td style="color:#8fa8bc;padding:6px 10px">Baseline — fast ML path, no LLM cost</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-pass">PASS</span></td><td style="color:#8fa8bc;padding:6px 10px;text-align:center">—</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">EC_002</td><td style="color:#8fa8bc;padding:6px 10px">E-commerce</td>
      <td style="color:#e8f0fe;padding:6px 10px">Large wire to Singapore (regular importer)</td>
      <td style="color:#8fa8bc;padding:6px 10px">LLM reads context, confirms legitimate corridor</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-monitor">MONITOR</span></td><td style="color:#4caf50;padding:6px 10px;text-align:center">✅</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">EC_003</td><td style="color:#8fa8bc;padding:6px 10px">E-commerce</td>
      <td style="color:#e8f0fe;padding:6px 10px">27× spike to Nigeria at 3AM</td>
      <td style="color:#8fa8bc;padding:6px 10px">Classic wire fraud — XGBoost catches it instantly</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-block">BLOCK</span></td><td style="color:#8fa8bc;padding:6px 10px;text-align:center">—</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">EC_004</td><td style="color:#8fa8bc;padding:6px 10px">E-commerce</td>
      <td style="color:#e8f0fe;padding:6px 10px">3.7× spike to Germany, 3AM</td>
      <td style="color:#8fa8bc;padding:6px 10px">Gray zone — LLM weighs risk vs. business context</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-review">REVIEW</span></td><td style="color:#4caf50;padding:6px 10px;text-align:center">✅</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">EC_005</td><td style="color:#8fa8bc;padding:6px 10px">E-commerce</td>
      <td style="color:#e8f0fe;padding:6px 10px">3× spike to HK, 10PM</td>
      <td style="color:#8fa8bc;padding:6px 10px">Edge case — allow with soft monitoring flag</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-monitor">MONITOR</span></td><td style="color:#4caf50;padding:6px 10px;text-align:center">✅</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.08)"><td colspan="6"></td></tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">IB_001</td><td style="color:#8fa8bc;padding:6px 10px">Invest. Banking</td>
      <td style="color:#e8f0fe;padding:6px 10px">Normal SWIFT to Deutsche Bank</td>
      <td style="color:#8fa8bc;padding:6px 10px">Known corridor, verified BIC, trade docs attached</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-pass">PASS</span></td><td style="color:#8fa8bc;padding:6px 10px;text-align:center">—</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">IB_002</td><td style="color:#8fa8bc;padding:6px 10px">Invest. Banking</td>
      <td style="color:#e8f0fe;padding:6px 10px">BEC: same supplier, new IBAN via email</td>
      <td style="color:#8fa8bc;padding:6px 10px">LLM identifies instruction-change pattern, flags for callback verification</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-review">REVIEW</span></td><td style="color:#4caf50;padding:6px 10px;text-align:center">✅</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">IB_003</td><td style="color:#8fa8bc;padding:6px 10px">Invest. Banking</td>
      <td style="color:#e8f0fe;padding:6px 10px">Wire to Iranian entity (OFAC listed)</td>
      <td style="color:#8fa8bc;padding:6px 10px">Pre-ML rule: OFAC hit = instant freeze, no model needed</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-block">BLOCK</span></td><td style="color:#8fa8bc;padding:6px 10px;text-align:center">—</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">IB_004</td><td style="color:#8fa8bc;padding:6px 10px">Invest. Banking</td>
      <td style="color:#e8f0fe;padding:6px 10px">New Cyprus counterparty, 11PM, no docs</td>
      <td style="color:#8fa8bc;padding:6px 10px">Multiple hard signals — XGBoost auto-blocks</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-block">BLOCK</span></td><td style="color:#8fa8bc;padding:6px 10px;text-align:center">—</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">IB_005</td><td style="color:#8fa8bc;padding:6px 10px">Invest. Banking</td>
      <td style="color:#e8f0fe;padding:6px 10px">BNP Paribas — same company, new IBAN registered</td>
      <td style="color:#8fa8bc;padding:6px 10px">Low-risk jurisdiction, moderate BEC flag — LLM recommends callback</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-monitor">MONITOR</span></td><td style="color:#4caf50;padding:6px 10px;text-align:center">✅</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.08)"><td colspan="6"></td></tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">PP_001</td><td style="color:#8fa8bc;padding:6px 10px">PayPal</td>
      <td style="color:#e8f0fe;padding:6px 10px">Normal P2P to Alex Johnson</td>
      <td style="color:#8fa8bc;padding:6px 10px">Trusted device, known recipient, domestic</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-pass">PASS</span></td><td style="color:#8fa8bc;padding:6px 10px;text-align:center">—</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">PP_002</td><td style="color:#8fa8bc;padding:6px 10px">PayPal</td>
      <td style="color:#e8f0fe;padding:6px 10px">Account Takeover — new device, 9 failed logins, IP from Russia</td>
      <td style="color:#8fa8bc;padding:6px 10px">Classic ATO pattern — XGBoost auto-blocks</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-block">BLOCK</span></td><td style="color:#8fa8bc;padding:6px 10px;text-align:center">—</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">PP_003</td><td style="color:#8fa8bc;padding:6px 10px">PayPal</td>
      <td style="color:#e8f0fe;padding:6px 10px">Velocity fraud — 35 transactions in 1 hour</td>
      <td style="color:#8fa8bc;padding:6px 10px">New recipients, low device trust — XGBoost auto-blocks</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-block">BLOCK</span></td><td style="color:#8fa8bc;padding:6px 10px;text-align:center">—</td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
      <td style="color:#e8f0fe;padding:6px 10px">PP_004</td><td style="color:#8fa8bc;padding:6px 10px">PayPal</td>
      <td style="color:#e8f0fe;padding:6px 10px">Known recipient but 5-day-old account + untrusted device</td>
      <td style="color:#8fa8bc;padding:6px 10px">Possible money mule — LLM recommends step-up auth</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-monitor">MONITOR</span></td><td style="color:#4caf50;padding:6px 10px;text-align:center">✅</td>
    </tr>
    <tr>
      <td style="color:#e8f0fe;padding:6px 10px">PP_005</td><td style="color:#8fa8bc;padding:6px 10px">PayPal</td>
      <td style="color:#e8f0fe;padding:6px 10px">New recipient + IP from China + 2 failed logins</td>
      <td style="color:#8fa8bc;padding:6px 10px">Mixed signals — LLM flags for manual review</td>
      <td style="padding:6px 10px;text-align:center"><span class="doc-badge badge-review">REVIEW</span></td><td style="color:#4caf50;padding:6px 10px;text-align:center">✅</td>
    </tr>
    </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Why LLM? ──────────────────────────────────────────────────
    st.markdown("### 🤖 Why DeepSeek-V3 (LLM)?")
    wl1, wl2 = st.columns(2)
    with wl1:
        st.markdown("""
        <div class="doc-card">
        <div class="doc-h2">What ML models can't do</div>
        <div style="color:#8fa8bc;font-size:0.84rem;line-height:1.7">
        ❌ Understand <b style="color:#e8f0fe">why</b> a user is making a large transfer<br>
        ❌ Know that 3AM in New York = business hours in Singapore<br>
        ❌ Distinguish BEC from a legitimate supplier account change<br>
        ❌ Generate a human-readable explanation for a compliance team<br>
        ❌ Write a non-accusatory email to a customer<br>
        ❌ Produce a checklist that references regulatory deadlines (SAR filing, BSA hold limits)
        </div>
        </div>""", unsafe_allow_html=True)
    with wl2:
        st.markdown("""
        <div class="doc-card">
        <div class="doc-h2">What DeepSeek-V3 adds</div>
        <div style="color:#8fa8bc;font-size:0.84rem;line-height:1.7">
        ✅ <b style="color:#e8f0fe">Context reasoning:</b> Reads the full user profile + transaction as narrative<br>
        ✅ <b style="color:#e8f0fe">Domain knowledge:</b> Knows OFAC, BSA, BEC patterns, SWIFT conventions<br>
        ✅ <b style="color:#e8f0fe">Risk calibration:</b> Weighs red flags against green flags holistically<br>
        ✅ <b style="color:#e8f0fe">Communication:</b> Generates customer-facing and ops-facing text simultaneously<br>
        ✅ <b style="color:#e8f0fe">Cost efficiency:</b> Only called for the ~15% of transactions ML is uncertain about
        </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;color:#7a95aa;font-size:0.75rem;margin-top:8px'>"
        "Tech stack: scikit-learn · XGBoost · DeepSeek-V3 via SiliconFlow API · "
        "Neon PostgreSQL · Streamlit</p>",
        unsafe_allow_html=True,
    )
