import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
import os

# 1. Load Files
MODEL_FILE = 'xgb_loan_model.json'
PKL_FILE = 'preprocessor.pkl'

model = xgb.XGBClassifier()
if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)

# Load preprocessor để lấy đúng StandardScaler và OneHot của Notebook
if os.path.exists(PKL_FILE):
    preprocessor = joblib.load(PKL_FILE)

def predict_loan(age, income, home_ownership, emp_length, intent, grade, amount, rate, percent_income, default):
    # 1. Tạo DataFrame đúng 10 cột gốc (Thứ tự phải giống y hệt Notebook)
    raw_df = pd.DataFrame([[
        age, income, home_ownership, emp_length, intent, 
        grade, amount, rate, percent_income, default
    ]], columns=[
        'person_age', 'person_income', 'person_home_ownership', 
        'person_emp_length', 'loan_intent', 'loan_grade', 
        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
        'cb_person_default_on_file'
    ])

    # 2. Transform dữ liệu qua Pipeline (QUAN TRỌNG NHẤT)
    # Bước này thay thế hoàn toàn việc tính toán manual mean/std
    input_processed = preprocessor.transform(raw_df)
    
    # Lấy tên cột sau khi encode để SHAP vẽ đúng tên
    feature_names = preprocessor.get_feature_names_out()
    input_df = pd.DataFrame(input_processed, columns=feature_names)

    # 3. Prediction Logic (Sửa lại cho đúng nhãn)
    prob_default = model.predict_proba(input_df)[0][1] # Xác suất nợ xấu (lớp 1)
    
    # Nếu xác suất nợ xấu > 50% => Rejected
    if prob_default > 0.5:
        prediction = "Rejected ❌"
    else:
        prediction = "Approved ✅"
    
    # 4. SHAP Plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    plt.figure(figsize=(10, 6))
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)): base_val = base_val[1]
    
    # Vẽ Waterfall Plot
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0], 
        base_values=base_val, 
        data=input_df.iloc[0], 
        feature_names=list(feature_names)
    ), show=False)
    
    plt.tight_layout()
    plot_path = "shap_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    return prediction, f"{prob_default:.2%}", plot_path

# 5. Giao diện Gradio
iface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Age", value=26),
        gr.Number(label="Annual Income ($)", value=44000),
        gr.Dropdown(["RENT", "OWN", "MORTGAGE", "OTHER"], label="Home Ownership", value="RENT"),
        gr.Number(label="Employment Length (years)", value=6),
        gr.Dropdown(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], label="Loan Intent", value="PERSONAL"),
        gr.Dropdown(["A", "B", "C", "D", "E", "F", "G"], label="Loan Grade", value="B"),
        gr.Number(label="Loan Amount ($)", value=11000),
        gr.Number(label="Interest Rate (%)", value=10.37),
        gr.Slider(0, 1, label="DTI Ratio", value=0.26),
        gr.Radio(["Y", "N"], label="Historical Default?", value="N")
    ],
    outputs=[
        gr.Textbox(label="Final Decision"),
        gr.Textbox(label="Probability of Default (Risk)"),
        gr.Image(label="SHAP Analysis")
    ],
    # title="🏦 AI Loan Approval System (Production Ready)",
    # description="This version uses the exact preprocessor from your Notebook to guarantee matching results."
    title="🏦 AI Loan Approval System",
    description="""
    ## 📘 Quick User Guide:
    * **🔴 Red bars:** Increase risk of **Rejection**.
    * **🔵 Blue bars:** Increase chance of **Approval**.
    * **Click on the rows below** to test typical Approved/Rejected cases.
    """,
    
    # === THÊM VÀO ĐÂY (VỊ TRÍ NÀY) ===
    examples=[
        [32, 60000, "MORTGAGE", 12, "VENTURE", "A", 5000, 7.5, 0.04, "N"], # Approved Case
        [26, 44000, "RENT", 6, "PERSONAL", "B", 11000, 10.37, 0.26, "N"]   # Rejected Case (Index 24851)
    ]
    # ================================
)

if __name__ == "__main__":
    iface.launch()