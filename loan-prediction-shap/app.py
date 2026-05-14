import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import joblib
import os

# ── Load model & preprocessor ────────────────────────────────────────────────
MODEL_FILE = 'xgb_loan_model.json'
PKL_FILE   = 'preprocessor.pkl'

model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)

if os.path.exists(PKL_FILE):
    preprocessor = joblib.load(PKL_FILE)


def predict_loan(age, income, home_ownership, emp_length, intent,
                 grade, amount, rate, percent_income, default):
    # Build raw input DataFrame (column order must match training data)
    raw_df = pd.DataFrame([[
        age, income, home_ownership, emp_length, intent,
        grade, amount, rate, percent_income, default
    ]], columns=[
        'person_age', 'person_income', 'person_home_ownership',
        'person_emp_length', 'loan_intent', 'loan_grade',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_default_on_file'
    ])

    # Preprocess through the saved pipeline
    input_processed = preprocessor.transform(raw_df)
    feature_names   = preprocessor.get_feature_names_out()
    input_df        = pd.DataFrame(input_processed, columns=feature_names)

    # class 1 = fully paid (approved), class 0 = default (rejected)
    prob_approve = float(model.predict_proba(input_df)[0][1])
    prob_default = 1.0 - prob_approve

    if prob_approve > 0.5:
        decision = "✅  LOAN APPROVED"
    else:
        decision = "❌  LOAN REJECTED"

    # ── SHAP Explanation ──────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # expected_value: scalar for newer SHAP, list for older versions
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1])

    # shap_values: 2-D array (n_samples, n_features) in new SHAP (positive class,
    # log-odds space), OR list [class0_array, class1_array] in older SHAP.
    # We always want the positive-class values for the first (only) sample.
    if isinstance(shap_values, list):
        sv = shap_values[1][0]   # class 1 → default/rejection risk, sample 0
    else:
        sv = shap_values[0]      # sample 0 (already positive-class in new SHAP)

    fig = plt.figure(figsize=(12, 7))
    shap.plots.waterfall(
        shap.Explanation(
            values=sv,
            base_values=base_val,
            data=input_df.iloc[0],
            feature_names=list(feature_names)
        ),
        show=False
    )
    plt.tight_layout()
    plot_path = "shap_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return (
        decision,
        f"{prob_default:.1%}",
        f"{prob_approve:.1%}",
        plot_path
    )


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
/* Hide Gradio footer */
footer { display: none !important; }

/* Decision output box */
#decision-box textarea {
    text-align: center !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    padding: 16px !important;
    border-radius: 10px !important;
    background: #f0fdf4 !important;
}

/* Submit button */
#submit-btn {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

/* Section divider */
.section-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 8px 0;
}
"""

# ── Gradio Blocks UI ──────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        font=gr.themes.GoogleFont("Inter")
    ),
    title="AI Loan Approval System",
    css=CSS
) as iface:

    # Header
    gr.HTML("""
        <div style="text-align:center; padding:28px 0 12px 0;">
            <h1 style="font-size:2.2rem; font-weight:800; margin:0; color:#1e1b4b;">
                🏦 AI Loan Approval System
            </h1>
            <p style="color:#6b7280; font-size:1rem; margin-top:8px;">
                Powered by <strong>XGBoost</strong> + <strong>SHAP</strong> —
                instant prediction with transparent, explainable AI
            </p>
        </div>
    """)

    # ── Input Section ──────────────────────────────────────────────────────────
    with gr.Row():
        # Left column — Applicant Info
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Applicant Information")
            age = gr.Number(
                label="Age (years)", value=26, minimum=18, maximum=100,
                info="Applicant's current age"
            )
            income = gr.Number(
                label="Annual Income ($)", value=44000,
                info="Gross yearly income before taxes"
            )
            emp_length = gr.Number(
                label="Employment Length (years)", value=6,
                minimum=0, maximum=50,
                info="Years at current employer"
            )
            home_ownership = gr.Dropdown(
                choices=["RENT", "OWN", "MORTGAGE", "OTHER"],
                label="Home Ownership", value="RENT"
            )
            default_history = gr.Radio(
                choices=["N", "Y"],
                label="Previous Default on File?", value="N",
                info="Has the applicant defaulted on a loan before?"
            )

        # Right column — Loan Details
        with gr.Column(scale=1):
            gr.Markdown("### 💳 Loan Details")
            amount = gr.Number(
                label="Loan Amount ($)", value=11000,
                info="Total amount requested"
            )
            rate = gr.Number(
                label="Interest Rate (%)", value=10.37,
                info="Annual percentage rate offered"
            )
            percent_income = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01,
                label="Debt-to-Income Ratio (DTI)", value=0.26,
                info="Loan installment ÷ annual income (0.26 = 26 %)"
            )
            intent = gr.Dropdown(
                choices=[
                    "PERSONAL", "EDUCATION", "MEDICAL",
                    "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
                ],
                label="Loan Purpose", value="PERSONAL"
            )
            grade = gr.Dropdown(
                choices=["A", "B", "C", "D", "E", "F", "G"],
                label="Loan Grade  (A = Best, G = Worst)", value="B",
                info="Credit grade assigned by the lending platform"
            )

    # Submit button
    submit_btn = gr.Button(
        "🔍  Analyze Loan Application",
        variant="primary", size="lg", elem_id="submit-btn"
    )

    gr.HTML('<hr class="section-divider"/>')

    # ── Output Section ─────────────────────────────────────────────────────────
    gr.Markdown("### 📊 Prediction Result")

    with gr.Row():
        decision_out = gr.Textbox(
            label="Final Decision",
            interactive=False,
            elem_id="decision-box",
            scale=2
        )
        with gr.Column(scale=1):
            prob_default_out = gr.Textbox(
                label="🔴 Default Risk",
                interactive=False
            )
            prob_approve_out = gr.Textbox(
                label="🟢 Approval Probability",
                interactive=False
            )

    gr.HTML('<hr class="section-divider"/>')

    # ── SHAP Section ───────────────────────────────────────────────────────────
    gr.Markdown("### 🔍 SHAP Explanation — Why this decision?")
    gr.Markdown(
        "> **🔴 Red bars** push toward **Rejection** (↑ default risk) &nbsp;·&nbsp; "
        "**🔵 Blue bars** push toward **Approval** (↓ default risk)  \n"
        "> The longer the bar, the stronger that feature's influence on the outcome."
    )
    shap_out = gr.Image(label="Feature Impact (SHAP Waterfall Plot)")

    gr.HTML('<hr class="section-divider"/>')

    # ── Examples ───────────────────────────────────────────────────────────────
    gr.Markdown("### 📋 Try Sample Applications — click a row to load")
    gr.Examples(
        examples=[
            # age, income, home, emp_len, intent, grade, amount, rate, dti, default
            [32, 60000, "MORTGAGE", 12, "VENTURE",  "A", 5000,  7.50, 0.04, "N"],  # Strong approval
            [26, 44000, "RENT",      6, "PERSONAL", "B", 11000, 10.37, 0.26, "N"],  # Borderline case
            [22, 25000, "RENT",      1, "PERSONAL", "E", 15000, 18.50, 0.60, "Y"],  # High-risk rejection
        ],
        inputs=[
            age, income, home_ownership, emp_length, intent,
            grade, amount, rate, percent_income, default_history
        ],
        label="Sample Applicants"
    )

    # ── Wire up button ─────────────────────────────────────────────────────────
    submit_btn.click(
        fn=predict_loan,
        inputs=[
            age, income, home_ownership, emp_length, intent,
            grade, amount, rate, percent_income, default_history
        ],
        outputs=[decision_out, prob_default_out, prob_approve_out, shap_out]
    )


if __name__ == "__main__":
    iface.launch()
