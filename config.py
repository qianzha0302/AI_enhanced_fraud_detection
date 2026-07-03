"""
Constants, thresholds, and LLM prompt templates for all three industry verticals.
"""

# ─── Stage thresholds (shared) ────────────────────────────────────────────────
IF_THRESHOLD          = -0.3
XGB_LOW_THRESHOLD     = 0.08
XGB_HIGH_THRESHOLD    = 0.985
XGB_REVIEW_THRESHOLD  = 0.50
XGB_MONITOR_THRESHOLD = 0.10

# ─── LLM ─────────────────────────────────────────────────────────────────────
LLM_MODEL    = "deepseek-ai/DeepSeek-V3"
LLM_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_TIMEOUT  = 30

# ─── Industries ───────────────────────────────────────────────────────────────
INDUSTRIES = ["ecommerce", "investment_banking", "payment_platform"]

INDUSTRY_LABELS = {
    "ecommerce":          "🛒 E-commerce",
    "investment_banking": "🏦 Investment Banking",
    "payment_platform":   "💳 Digital Payment Platform",
}

# ─── Feature columns per industry ────────────────────────────────────────────
# IEEE-CIS Fraud Detection dataset features (all <20% missing, NaN filled with 0)
ECOMMERCE_FEATURES = [
    "TransactionAmt", "TransactionDT",
    "card1", "card2", "card3", "card5",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D10", "D15",
    "V15", "V16", "V17", "V18", "V33", "V34",
    "V57", "V58", "V73", "V74",
    "V79", "V80", "V81", "V84", "V86", "V87",
    "V92", "V93", "V94",
]

# IBM Transactions for Anti Money Laundering (AML) — HI-Small dataset
# All features are derived from raw CSV columns; no external enrichment needed.
IB_FEATURES = [
    "log_amount_paid",      # log1p(Amount Paid)  — amount spans $1e-6 to $1e12
    "log_amount_received",  # log1p(Amount Received)
    "amount_ratio",         # Amount Received / Amount Paid (always 1.0 for laundering)
    "hour",                 # Hour of day extracted from Timestamp
    "day_of_week",          # 0=Monday … 6=Sunday
    "is_cross_currency",    # Receiving Currency != Payment Currency (0% fraud if cross)
    "same_bank",            # From Bank == To Bank
    "same_account",         # Account == Account.1 (self-transfer)
    "is_ach",               # Payment Format = ACH  (86% of all laundering cases)
    "is_bitcoin",           # Payment Format = Bitcoin
    "is_cash",              # Payment Format = Cash
    "is_cheque",            # Payment Format = Cheque
    "is_credit_card",       # Payment Format = Credit Card
    "is_wire",              # Payment Format = Wire  (0% laundering)
    "is_reinvestment",      # Payment Format = Reinvestment (0% laundering)
]

# Real creditcard.csv PCA features — trained on ULB Credit Card Fraud dataset (284K rows)
CREDITCARD_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Payment platform uses real creditcard PCA features for ML scoring;
# domain fields (device_trust_score, ip_country, etc.) are kept in transaction
# dicts for LLM display only and are not used by the ML model.
PAYMENT_PLATFORM_FEATURES = CREDITCARD_FEATURES

INDUSTRY_FEATURES = {
    "ecommerce":          ECOMMERCE_FEATURES,
    "investment_banking": IB_FEATURES,
    "payment_platform":   PAYMENT_PLATFORM_FEATURES,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Context analysis prompts (one per industry)
# ═══════════════════════════════════════════════════════════════════════════════

CONTEXT_ANALYSIS_PROMPT = {

"ecommerce": """You are a fraud detection expert at a major financial institution.
Analyze the following e-commerce transaction and provide a structured risk assessment.

USER PROFILE:
- Account age: {account_age_days} days
- Typical transaction: ${user_avg_amount:.0f} (±${user_std:.0f})
- Frequency: {user_transaction_frequency}/month | Intl wires: {user_international_frequency}
- Industry: {user_industry} | Location: {user_primary_location}
- Fraud history: {user_fraud_history} | Dispute rate: {user_dispute_rate:.1f}%

CURRENT TRANSACTION:
- Amount: ${amount:.0f} (×{amount_deviation:.1f} vs user average)
- Merchant: {merchant_name} ({merchant_category})
- Destination: {destination_country} | First time to country: {is_new_country}
- Time: {transaction_time_hour}:00 (user typical: {user_typical_hour}:00)
- Method: {payment_method}

MODEL SCORES:
- IF anomaly score: {if_score:.3f} | XGBoost fraud prob: {fraud_prob:.1%}

Respond in this EXACT format:
🔴 Red Flags:
[numbered list]
🟢 Green Flags:
[numbered list]
🟡 Needs Verification:
[numbered list]
【Assessment】
[1-2 sentences: risk judgment + recommended action]""",


"investment_banking": """You are a Senior Compliance Officer at a Tier-1 investment bank.
Analyze this wire transfer for fraud, BEC, AML, and sanctions risk.

CLIENT PROFILE:
- Account age: {account_age_days} days
- Monthly wire volume: ${monthly_wire_volume:,.0f}
- Industry: {client_industry} | AML risk tier: {aml_risk_tier}
- Dispute rate: {user_dispute_rate:.1f}% | Relationship manager: {relationship_manager}

WIRE DETAILS:
- Amount: ${amount:,.0f} (×{amount_vs_monthly_avg:.1f}x monthly average)
- Counterparty: {counterparty_name} | Bank: {counterparty_bank}
- Counterparty country: {counterparty_country} | Risk score: {counterparty_risk_score:.2f}/1.0
- SWIFT BIC verified: {swift_verified} | OFAC/sanctions hit: {ofac_hit}
- Payment purpose: {payment_purpose}
- Trade docs attached: {trade_doc_present}
- BEC risk score: {bec_risk_score:.2f}/1.0 (email domain age, instruction change recency)
- First wire to this counterparty: {is_new_counterparty}
- Initiated at: {transaction_time_hour}:00

MODEL SCORES:
- IF anomaly score: {if_score:.3f} | XGBoost fraud prob: {fraud_prob:.1%}

Assess using these frameworks: BEC/Invoice Fraud, OFAC Sanctions, AML Red Flags, Account Takeover.
Respond in this EXACT format:
🔴 Red Flags:
[numbered list — cite specific values]
🟢 Green Flags:
[numbered list]
🟡 Compliance Verification Required:
[numbered list — specific docs or checks needed]
【Assessment】
[Risk classification (LOW/MEDIUM/HIGH) + recommended action + regulatory basis if applicable]""",


"payment_platform": """You are a Fraud Analyst at a large consumer payment platform.
Analyze this payment for account takeover, velocity fraud, and money mule risk.

ACCOUNT PROFILE:
- Account age: {account_age_days} days
- Typical payment: ${user_avg_amount:.0f} (±${user_std:.0f})
- Dispute rate: {user_dispute_rate:.1f}% | Verified: {account_verified}
- Primary country: {user_primary_country}

CURRENT PAYMENT:
- Amount: ${amount:.0f} (×{amount_deviation:.1f}x typical)
- Recipient: {recipient_name} | Recipient acct age: {recipient_account_age_days} days
- New recipient: {new_recipient}
- Device: {'NEW — unrecognised' if new_device else 'trusted device'} | Trust score: {device_trust_score:.2f}/1.0
- IP country: {ip_country} | Mismatch with account country: {ip_country_mismatch}
- Failed logins in last 24h: {failed_logins_24h}
- Transactions in last hour: {transactions_last_hour} (velocity signal)

MODEL SCORES:
- IF anomaly score: {if_score:.3f} | XGBoost fraud prob: {fraud_prob:.1%}

Assess using: Account Takeover (ATO), Velocity Fraud, Money Mule, First-party Fraud.
Respond in this EXACT format:
🔴 Red Flags:
[numbered list]
🟢 Green Flags:
[numbered list]
🟡 Needs Verification:
[numbered list]
【Assessment】
[Risk type + recommended action (e.g. step-up auth, freeze, monitor)]""",

}

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4a — Customer-facing messages per industry
# ═══════════════════════════════════════════════════════════════════════════════

CUSTOMER_EMAIL_PROMPT = {

"ecommerce": """Write a brief, friendly, non-accusatory verification email from the SecureBank Security Team.
Transaction: ${amount:.0f} to {merchant_name} in {destination_country}
Customer reference ID: {user_id}
Key concerns: {red_flags_summary}

STRICT RULES — violating any of these fails the task:
- Open with "Dear Valued Customer" — NEVER use placeholder brackets like [Name] or [Customer]
- Sign as "SecureBank Security Team | 1-800-555-0199"
- Protect-not-accuse tone, 2-3 specific reasons, 2 easy steps to verify
- Under 120 words total
Output ONLY Subject line + email body. No preamble, no explanation.""",

"investment_banking": """Write a professional wire verification notice from BNY Compliance Operations.
Wire: ${amount:,.0f} to {counterparty_name} ({counterparty_country})
Reference: {transaction_id}
Concerns: {red_flags_summary}

STRICT RULES — violating any of these fails the task:
- Open with "Dear Client," — NEVER use placeholder brackets like [Name] or [Client Name]
- Sign as "BNY Compliance Operations | +1 212-495-1784"
- Formal tone, reference regulatory obligations briefly
- Request original invoice/contract + beneficial owner confirmation
- State 4-hour review SLA
- Under 150 words total
Output ONLY Subject line + email body. No preamble, no explanation.""",

"payment_platform": """Write a concise security alert from PayPal Trust & Safety.
Payment: ${amount:.0f} to {recipient_name}
Concern: {red_flags_summary}

STRICT RULES — violating any of these fails the task:
- Open with "Hi there," — NEVER use placeholder brackets like [Name] or [User]
- Sign as "PayPal Trust & Safety | paypal.com/help or 1-888-221-1161"
- Urgent but not accusatory tone
- Mention unusual sign-in activity detected
- Offer SMS step-up verification
- Under 100 words total
Output ONLY Subject line + email body. No preamble, no explanation.""",

}

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4b — Ops action items per industry
# ═══════════════════════════════════════════════════════════════════════════════

OPS_ACTIONS_PROMPT = {

"ecommerce": """Generate an ops checklist for a flagged e-commerce transaction.
Transaction: ${amount:.0f} to {merchant_name} in {destination_country} | Decision: {decision}
Context: {llm_analysis_summary}

Format with sections: 📧 CUSTOMER OUTREACH / 📋 REQUIRED DOCS / ✅ VERIFICATION CRITERIA / ⏱️ TIMELINE / 🚨 ESCALATION
Use □ checkboxes. Under 150 words.""",

"investment_banking": """Generate a Compliance ops checklist for a flagged wire transfer.
Wire: ${amount:,.0f} to {counterparty_name} | Decision: {decision}
Context: {llm_analysis_summary}

Format with sections: 📞 CLIENT OUTREACH / 📋 REQUIRED DOCUMENTS / 🔍 COMPLIANCE CHECKS / ⏱️ REGULATORY TIMELINE / 🚨 ESCALATION (include SAR filing trigger)
Use □ checkboxes. Under 180 words.""",

"payment_platform": """Generate a Trust & Safety ops checklist for a flagged PayPal payment.
Payment: ${amount:.0f} to {recipient_name} | Decision: {decision}
Context: {llm_analysis_summary}

Format with sections: 🔐 ACCOUNT ACTION / 📧 USER NOTIFICATION / 🔍 INVESTIGATION STEPS / ⏱️ TIMELINE / 🚨 ESCALATION
Use □ checkboxes. Under 130 words.""",

}

# ═══════════════════════════════════════════════════════════════════════════════
# Mock responses (used when no API key)
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_CONTEXT_ANALYSIS = {

"ecommerce": """🔴 Red Flags:
1. Amount ${amount:.0f} is {amount_deviation:.0f}x the user's typical transaction
2. {destination_country} flagged as elevated-risk jurisdiction
3. First transaction to this country — no prior corridor history
4. Initiated at {transaction_time_hour}:00 — outside user's typical banking hours

🟢 Green Flags:
1. Account age: {account_age_days} days — long-standing customer
2. Zero dispute history ({user_dispute_rate:.1f}% vs 3% industry avg)
3. Merchant category ({merchant_category}) aligns with user's industry ({user_industry})

🟡 Needs Verification:
1. Is this amount proportional to a real invoice or contract?
2. Can the customer provide a purchase order or trade agreement?
3. Why was the transaction initiated outside normal hours?

【Assessment】
Medium risk. Likely legitimate but the combination of new geography, elevated amount, and unusual timing warrants a quick document check before releasing funds.""",

"investment_banking": """🔴 Red Flags:
1. Amount ${amount:,.0f} is {amount_vs_monthly_avg:.1f}x client's monthly average volume
2. BEC risk score {bec_risk_score:.2f}/1.0 — payment instructions changed recently
3. First wire to this counterparty ({counterparty_name}) with no prior relationship
4. {counterparty_country} jurisdiction — elevated AML risk score {counterparty_risk_score:.2f}/1.0
5. Wire initiated at {transaction_time_hour}:00 — outside standard banking hours

🟢 Green Flags:
1. SWIFT BIC verified: {swift_verified}
2. Client account age {account_age_days} days — established relationship
3. No OFAC/sanctions hit: {ofac_hit}

🟡 Compliance Verification Required:
1. Request original invoice/contract (not email attachment) via secure channel
2. Verify payment instructions directly with counterparty CFO by phone
3. Confirm beneficial ownership of receiving entity
4. Review email chain for signs of domain spoofing

【Assessment】
HIGH risk — BEC/Invoice Fraud pattern. Wire must be held. Initiate callback verification procedure per BSA/AML policy. If unverified within 4 hours, consider filing a precautionary SAR.""",

"payment_platform": """🔴 Red Flags:
1. New unrecognised device with trust score {device_trust_score:.2f}/1.0
2. {failed_logins_24h} failed login attempts in the past 24 hours — ATO signal
3. IP country mismatch: account registered in US, login from {ip_country}
4. {transactions_last_hour} transactions in last hour — velocity anomaly
5. New recipient with account age {recipient_account_age_days} days

🟢 Green Flags:
1. Account age {account_age_days} days — established account
2. Payment amount ${amount:.0f} within historical range
3. Account dispute rate {user_dispute_rate:.1f}%

🟡 Needs Verification:
1. Step-up authentication: send SMS OTP to registered phone number
2. Confirm recipient identity (known contact?)
3. Review recent login history for pattern

【Assessment】
HIGH risk — Account Takeover (ATO) pattern. Suspend session immediately. Require step-up auth before any fund movement. Freeze account if user cannot verify via registered device/phone.""",

}

MOCK_CUSTOMER_EMAIL = {

"ecommerce": """Subject: Action Required — Verify Your Recent Wire Transfer

Dear Valued Customer,

We paused a ${amount:.0f} wire to {merchant_name} ({destination_country}) to make sure it's really you.

Why we paused it:
• ${amount:.0f} is larger than your typical transactions
• First wire to {destination_country}
• Transfer initiated outside your usual hours

To release the funds (5 minutes):
1. Reply with a copy of the invoice or contract
2. Or call 1-800-BANK-NOW to confirm verbally

We'll review within 4 business hours.

Best regards, Security Team""",

"investment_banking": """Subject: Wire Transfer Verification Required — Ref #{transaction_id}

Dear Client,

As part of our regulatory compliance obligations, we are placing a temporary hold on the following wire transfer pending verification:

Amount: ${amount:,.0f}
Beneficiary: {counterparty_name}
Destination: {counterparty_country}

Required documentation (please provide within 4 hours):
1. Original commercial invoice or contract (via secure portal)
2. Beneficial ownership confirmation for receiving entity
3. Brief description of the business purpose

Our Compliance team will review within 4 business hours. Verified wires are released within 30 minutes of approval.

Please contact your Relationship Manager directly if you have questions.

Regards,
Wire Compliance Team""",

"payment_platform": """Subject: Unusual Activity Detected — Action Required

Hi there,

We detected unusual sign-in activity on your account and temporarily paused your ${amount:.0f} payment to {recipient_name}.

To confirm it's you, please verify your identity:
1. Open PayPal on your trusted device
2. Enter the SMS code we just sent to your registered phone

This takes less than 2 minutes. Your payment will be released immediately after.

If you didn't initiate this activity, please call 1-888-PAYPAL-1 immediately.

Stay safe,
PayPal Trust & Safety""",

}

MOCK_OPS_ACTIONS = {

"ecommerce": """📧 CUSTOMER OUTREACH
□ Send verification email immediately
□ SMS: "Check email — wire transfer needs quick confirmation"
□ Set 24h callback reminder

📋 REQUIRED DOCUMENTATION
□ Commercial invoice or proforma invoice
□ Purchase order or contract
□ Brief explanation of wire timing

✅ VERIFICATION CRITERIA (any 2 of 3)
□ Valid invoice matching wire amount
□ Confirmed business relationship with merchant
□ Customer phone confirmation

⏱️ TIMELINE
• Response deadline: 24h | Review SLA: 4h | Auto-release: 30min after approval

🚨 ESCALATION
→ 12h no response: Call account manager
→ 24h no response: Auto-decline with explanation""",

"investment_banking": """📞 CLIENT OUTREACH
□ Call Relationship Manager within 30 minutes
□ Send secure portal notification with doc checklist
□ Log hold in wire tracking system (Case #{transaction_id})

📋 REQUIRED DOCUMENTS
□ Original invoice/contract (not email attachment)
□ Beneficial ownership declaration (UBO form)
□ Payment purpose statement signed by authorized signatory
□ Callback verification: confirm instructions with counterparty CFO directly

🔍 COMPLIANCE CHECKS
□ Re-run OFAC/sanctions screen on beneficiary + UBOs
□ Check counterparty against PEP/adverse media database
□ Review email chain for domain spoofing indicators

⏱️ REGULATORY TIMELINE
• Client response: 4h | Compliance review: 2h | Release: 30min after approval
• BSA hold limit: 5 business days maximum

🚨 ESCALATION
→ 2h no response: Escalate to Chief Compliance Officer
→ BEC confirmed: File SAR within 30 days (FinCEN requirement)
→ OFAC hit: Freeze immediately, notify Legal""",

"payment_platform": """🔐 ACCOUNT ACTION
□ Suspend current session immediately
□ Require step-up authentication (SMS OTP to registered phone)
□ Block payment until identity confirmed

📧 USER NOTIFICATION
□ Send security alert email + SMS
□ Push notification if mobile app installed
□ Log incident in fraud case management system

🔍 INVESTIGATION STEPS
□ Review last 10 login IPs and device fingerprints
□ Check if registered phone/email were recently changed
□ Verify recipient account for money mule patterns

⏱️ TIMELINE
• Step-up auth: 15min window | Account review: 2h | Escalate if no response: 4h

🚨 ESCALATION
→ ATO confirmed: Freeze account, reverse any unauthorised payments (up to 72h)
→ Notify user via backup contact method
→ If money mule: File SAR, coordinate with law enforcement""",

}
