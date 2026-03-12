import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0c0f;
    --surface:   #111318;
    --border:    #1e2330;
    --accent:    #e8584a;
    --accent2:   #f0a57a;
    --safe:      #4ade80;
    --text:      #e8e8e8;
    --muted:     #6b7280;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(232,88,74,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(240,165,122,0.05) 0%, transparent 60%),
        var(--bg) !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }

/* Hide sidebar toggle */
[data-testid="collapsedControl"] { display: none; }

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    background: #d44a3c !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(232,88,74,0.3) !important;
}

/* Select boxes & number inputs */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 2rem !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Card style */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
}

.risk-high {
    background: linear-gradient(135deg, rgba(232,88,74,0.15), rgba(232,88,74,0.05));
    border: 1px solid rgba(232,88,74,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, rgba(74,222,128,0.12), rgba(74,222,128,0.04));
    border: 1px solid rgba(74,222,128,0.35);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.risk-mid {
    background: linear-gradient(135deg, rgba(240,165,122,0.12), rgba(240,165,122,0.04));
    border: 1px solid rgba(240,165,122,0.35);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.badge {
    display: inline-block;
    background: rgba(232,88,74,0.15);
    border: 1px solid rgba(232,88,74,0.3);
    color: var(--accent2);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.label {
    color: var(--muted);
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* Progress bar for risk gauge */
.gauge-wrap { margin: 1rem 0; }
.gauge-track {
    height: 10px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s ease;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('best_model_v1_realmlp.pkl', 'rb') as f:
        model = pickle.load(f)
    # Force CPU — Streamlit Cloud has no GPU
    try:
        model.alg_interface_.device = 'cpu'
        model.alg_interface_.to('cpu')
        if hasattr(model.alg_interface_, 'model') and model.alg_interface_.model is not None:
            model.alg_interface_.model = model.alg_interface_.model.cpu()
    except Exception:
        pass
    return model

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False




# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 1.5rem 0;">
    <span class="badge">Hack4Health 2026</span>
    <h1 style="font-size: 3.2rem; margin: 0.6rem 0 0.2rem 0; line-height: 1.1;">
        CardioSense <span style="color:#e8584a;">AI</span>
    </h1>
    <p style="color:#6b7280; font-size:1rem; margin:0; max-width:520px;">
        Early cardiovascular disease risk detection powered by machine learning —
        trained on 70,000 de-identified patient records.
    </p>
</div>
<hr/>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model file `best_model_v1_realmlp.pkl` not found. Place it in the same directory as this app.")
    st.stop()


# ── Layout: Input | Results ───────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("<p class='label'>Patient Information</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age_years = st.slider("Age (years)", 20, 80, 45)
        age = (age_years * 365.25) / 23713  # raw days / max days in dataset

        height = st.slider("Height (cm)", 145, 190, 165)
        ap_hi  = st.slider("Systolic BP (ap_hi)", 80, 180, 120)
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3],
                                   format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above"}[x])
        smoke  = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")
        active = st.selectbox("Physically Active", [1, 0], format_func=lambda x: "Yes" if x else "No")

    with c2:
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        weight = st.slider("Weight (kg)", 45, 105, 70)
        ap_lo  = st.slider("Diastolic BP (ap_lo)", 60, 120, 80)
        gluc   = st.selectbox("Glucose", [1, 2, 3],
                               format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above"}[x])
        alco   = st.selectbox("Alcohol intake", [0, 1], format_func=lambda x: "Yes" if x else "No")

    bmi = weight / (height / 100) ** 2
    pulse_pressure = ap_hi - ap_lo

    st.markdown(f"""
    <div class="card" style="margin-top:1rem;">
        <p class="label">Computed values</p>
        <div style="display:flex; gap:2rem;">
            <div>
                <div style="font-family:'DM Mono',monospace; font-size:1.5rem; color:#e8e8e8;">{bmi:.1f}</div>
                <div style="color:#6b7280; font-size:0.8rem;">BMI</div>
            </div>
            <div>
                <div style="font-family:'DM Mono',monospace; font-size:1.5rem; color:#e8e8e8;">{pulse_pressure}</div>
                <div style="color:#6b7280; font-size:0.8rem;">Pulse Pressure</div>
            </div>
            <div>
                <div style="font-family:'DM Mono',monospace; font-size:1.5rem; color:#e8e8e8;">{int(age_years * 365.25)}</div>
                <div style="color:#6b7280; font-size:0.8rem;">Age (days)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("🫀 Analyse Risk", use_container_width=True)


with right:
    st.markdown("<p class='label'>Risk Assessment</p>", unsafe_allow_html=True)

    if predict_btn:
        # v3 digit decomposition — computed from raw inputs
        age_days = int(age_years * 365.25)

        input_data = pd.DataFrame([{
            # Original features
            'age':        age,
            'gender':     gender,
            'height':     height,
            'weight':     weight,
            'ap_hi':      ap_hi,
            'ap_lo':      ap_lo,
            'cholesterol': cholesterol,
            'gluc':       gluc,
            'smoke':      smoke,
            'alco':       alco,
            'active':     active,
            # ap_hi digits
            'ap_hi_hundreds': ap_hi // 100,
            'ap_hi_tens':     ap_hi % 100 // 10,
            'ap_hi_units':    ap_hi % 10,
            # ap_lo digits
            'ap_lo_hundreds': ap_lo // 100,
            'ap_lo_tens':     ap_lo % 100 // 10,
            'ap_lo_units':    ap_lo % 10,
            # height digits
            'height_hundreds': height // 100,
            'height_tens':     height % 100 // 10,
            'height_units':    height % 10,
            # weight digits
            'weight_tens':  int(weight) // 10,
            'weight_units': int(weight) % 10,
            # age digits
            'age_d1': int(age * 10) % 10,
            'age_d2': int(age * 100) % 10,
        }])

        prob = model.predict_proba(input_data)[0][1]

        # Manual adjustment for smoke/alcohol — these have near-zero SHAP
        # in this dataset due to self-reporting bias (only 8.7% smokers)
        # Adding clinically-informed adjustment on top of model output
        if smoke == 1:
            prob = min(1.0, prob + 0.05)
        if alco == 1:
            prob = min(1.0, prob + 0.03)

        pct  = prob * 100

        if pct >= 60:
            risk_label, risk_class, risk_color, emoji = "HIGH RISK", "risk-high", "#e8584a", "🔴"
        elif pct >= 40:
            risk_label, risk_class, risk_color, emoji = "MODERATE RISK", "risk-mid", "#f0a57a", "🟡"
        else:
            risk_label, risk_class, risk_color, emoji = "LOW RISK", "risk-low", "#4ade80", "🟢"

        gauge_color = risk_color

        st.markdown(f"""
        <div class="{risk_class}">
            <div style="font-size:3rem; margin-bottom:0.3rem;">{emoji}</div>
            <div style="font-family:'DM Serif Display',serif; font-size:2.2rem; color:{risk_color};">
                {pct:.1f}%
            </div>
            <div style="font-family:'DM Mono',monospace; font-size:0.85rem; letter-spacing:0.12em;
                        color:{risk_color}; margin-top:0.3rem;">
                {risk_label}
            </div>
            <div style="color:#6b7280; font-size:0.82rem; margin-top:0.8rem;">
                CVD probability based on your clinical profile
            </div>
        </div>

        <div class="gauge-wrap" style="margin-top:1.2rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                <span style="color:#6b7280; font-size:0.75rem; font-family:'DM Mono',monospace;">LOW</span>
                <span style="color:#6b7280; font-size:0.75rem; font-family:'DM Mono',monospace;">HIGH</span>
            </div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{pct}%;
                    background: linear-gradient(90deg, #4ade80, #f0a57a, #e8584a);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key factors
        st.markdown("<br/><p class='label'>Key Risk Factors Detected</p>", unsafe_allow_html=True)

        flags = []
        if ap_hi >= 140:          flags.append(("🔺 High Systolic BP", f"{ap_hi} mmHg"))
        if bmi >= 30:             flags.append(("🔺 Obesity", f"BMI {bmi:.1f}"))
        if cholesterol >= 2:      flags.append(("🔺 Elevated Cholesterol", f"Level {cholesterol}"))
        if age_years >= 55:       flags.append(("🔺 Age Risk", f"{age_years} years"))
        if smoke == 1:            flags.append(("🚬 Smoker", "Active"))
        if active == 0:           flags.append(("⚠️ Sedentary", "No exercise"))
        if not flags:             flags.append(("✅ Profile looks healthy", "Keep it up!"))

        cols = st.columns(2)
        for i, (label, val) in enumerate(flags):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background:var(--surface); border:1px solid var(--border);
                            border-radius:10px; padding:0.7rem 1rem; margin-bottom:0.5rem;">
                    <div style="font-size:0.82rem; color:#e8e8e8;">{label}</div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.75rem;
                                color:#6b7280; margin-top:0.2rem;">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        # SHAP section
        st.markdown("<br/><p class='label'>Model Explanation (SHAP)</p>", unsafe_allow_html=True)
        try:
            shap_vals  = np.load('shap_values.npy')        # shape: (n_samples, n_features)
            X_test_df  = pd.read_csv('X_test.csv')

            # Mean absolute SHAP per feature across all test samples
            mean_shap  = np.abs(shap_vals).mean(axis=0)
            importance = dict(zip(X_test_df.columns, mean_shap))

            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:6]
            max_val    = max(v for _, v in sorted_imp) or 1

            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor('#111318')
            ax.set_facecolor('#111318')

            names  = [k for k, _ in sorted_imp][::-1]
            scores = [v for _, v in sorted_imp][::-1]
            colors = ['#e8584a' if s > max_val * 0.6 else '#f0a57a' if s > max_val * 0.3 else '#6b7280'
                      for s in scores]

            ax.barh(names, scores, color=colors, height=0.55)
            ax.set_xlim(0, max_val * 1.2)
            ax.tick_params(colors='#9ca3af', labelsize=9)
            ax.set_xlabel("Mean |SHAP value|", color='#6b7280', fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e2330')
            ax.xaxis.label.set_color('#6b7280')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        except FileNotFoundError:
            st.info("ℹ️ Place `shap_values.npy` and `X_test.csv` in the same folder to enable SHAP.")

    else:
        st.markdown("""
        <div style="height:400px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center; text-align:center;
                    border: 1px dashed #1e2330; border-radius:16px;">
            <div style="font-size:4rem; margin-bottom:1rem;">🫀</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.4rem;
                        color:#e8e8e8; margin-bottom:0.5rem;">
                Ready to Analyse
            </div>
            <div style="color:#6b7280; font-size:0.88rem; max-width:260px; line-height:1.5;">
                Fill in the patient details on the left and click <em>Analyse Risk</em>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            padding: 0.5rem 0 1.5rem 0;">
    <div style="color:#6b7280; font-size:0.78rem; font-family:'DM Mono',monospace;">
        CardioSense AI · Hack4Health 2026 · Model AUC 0.800
    </div>
    <div style="color:#6b7280; font-size:0.78rem; font-family:'DM Mono',monospace;">
        ⚠️ Not a substitute for medical advice
    </div>
</div>
""", unsafe_allow_html=True)