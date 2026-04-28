import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

# 1. Tên file mô hình
MODEL_FILE = 'xgb_loan_model.json' 

# 2. Khởi tạo và nạp trọng số
model = xgb.XGBClassifier()

if os.path.exists(MODEL_FILE):
    model.load_model(MODEL_FILE)
    print(f"Successfully loaded {MODEL_FILE}")
else:
    print(f"Error: {MODEL_FILE} not found!")

# 3. Danh sách 25 cột mà model yêu cầu (COPY TỪ NOTEBOOK CỦA BẠN)
EXPECTED_COLUMNS = [
    'num__person_age', 'num__person_income', 'num__person_emp_length', 
    'num__loan_amnt', 'num__loan_int_rate', 'num__loan_percent_income', 
    'cat__person_home_ownership_MORTGAGE', 'cat__person_home_ownership_OTHER', 
    'cat__person_home_ownership_OWN', 'cat__person_home_ownership_RENT', 
    'cat__loan_intent_DEBTCONSOLIDATION', 'cat__loan_intent_EDUCATION', 
    'cat__loan_intent_HOMEIMPROVEMENT', 'cat__loan_intent_MEDICAL', 
    'cat__loan_intent_PERSONAL', 'cat__loan_intent_VENTURE', 
    'cat__loan_grade_A', 'cat__loan_grade_B', 'cat__loan_grade_C', 
    'cat__loan_grade_D', 'cat__loan_grade_E', 'cat__loan_grade_F', 
    'cat__loan_grade_G', 'cat__cb_person_default_on_file_N', 
    'cat__cb_person_default_on_file_Y'
]

def predict_loan(age, income, home_ownership, emp_length, intent, grade, amount, rate, percent_income, default, cred_hist):
    # Tạo DataFrame trống với 25 cột, giá trị mặc định là 0
    input_df = pd.DataFrame(0, index=[0], columns=EXPECTED_COLUMNS)

    # Đổ dữ liệu vào các cột Numerical
    input_df['num__person_age'] = age
    input_df['num__person_income'] = income
    input_df['num__person_emp_length'] = emp_length
    input_df['num__loan_amnt'] = amount
    input_df['num__loan_int_rate'] = rate
    input_df['num__loan_percent_income'] = percent_income
    # Lưu ý: cb_person_cred_hist_length không có trong danh sách 25 cột bạn gửi nên mình tạm bỏ qua

    # Đổ dữ liệu vào các cột Categorical (One-Hot Encoding)
    if f'cat__person_home_ownership_{home_ownership}' in EXPECTED_COLUMNS:
        input_df[f'cat__person_home_ownership_{home_ownership}'] = 1
    
    if f'cat__loan_intent_{intent}' in EXPECTED_COLUMNS:
        input_df[f'cat__loan_intent_{intent}'] = 1
        
    if f'cat__loan_grade_{grade}' in EXPECTED_COLUMNS:
        input_df[f'cat__loan_grade_{grade}'] = 1
        
    if f'cat__cb_person_default_on_file_{default}' in EXPECTED_COLUMNS:
        input_df[f'cat__cb_person_default_on_file_{default}'] = 1

    # Dự báo
    prob = model.predict_proba(input_df)[0][1]
    prediction = "Approved" if prob > 0.5 else "Rejected"
    
    # Giải thích bằng SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    plt.figure(figsize=(10, 6))
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    shap.plots._waterfall.waterfall_legacy(
        expected_value,
        shap_values[0],
        feature_names=EXPECTED_COLUMNS,
        show=False
    )
    
    plt.tight_layout()
    plot_path = "shap_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    return prediction, f"{prob:.2%}", plot_path

# 4. Giao diện Gradio
iface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Age", value=25),
        gr.Number(label="Annual Income", value=50000),
        gr.Dropdown(["RENT", "OWN", "MORTGAGE", "OTHER"], label="Home Ownership"),
        gr.Number(label="Employment Length (years)", value=2),
        gr.Dropdown(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], label="Loan Intent"),
        gr.Dropdown(["A", "B", "C", "D", "E", "F", "G"], label="Loan Grade"),
        gr.Number(label="Loan Amount", value=10000),
        gr.Number(label="Interest Rate (%)", value=11.0),
        gr.Slider(0, 1, label="Debt-to-Income Ratio", value=0.2),
        gr.Radio(["Y", "N"], label="Historical Default?"),
        gr.Number(label="Credit History Length (years)", value=5)
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Textbox(label="Approval Probability"),
        gr.Image(label="Model Explanation (SHAP Waterfall Plot)")
    ],
    title="Loan Approval Prediction System",
    description="Professional Loan Approval Analysis with SHAP Explanations."
)

if __name__ == "__main__":
    iface.launch()