import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import os

# ── Load model & preprocessor ─────────────────────────────────────────────────
model = xgb.XGBClassifier()
if os.path.exists('xgb_loan_model.json'):
    model.load_model('xgb_loan_model.json')
preprocessor = joblib.load('preprocessor.pkl') if os.path.exists('preprocessor.pkl') else None

# ── Design tokens ─────────────────────────────────────────────────────────────
C_GREEN  = '#10b981'
C_RED    = '#ef4444'
C_BLUE   = '#3b82f6'
C_GOLD   = '#e8b84b'
C_PURPLE = '#8b5cf6'
C_TEAL   = '#14b8a6'
C_BG     = '#161b22'
C_CARD   = '#1c2333'
C_BORDER = '#30363d'
C_TEXT   = '#e5e7eb'
C_MUTED  = '#9ca3af'

_L = dict(
    paper_bgcolor=C_BG, plot_bgcolor=C_BG,
    font=dict(color=C_TEXT, family='Inter, sans-serif', size=12),
    margin=dict(l=48, r=32, t=52, b=40),
    xaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER),
    yaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER),
)
_LEGEND = dict(bgcolor='rgba(0,0,0,0)', bordercolor=C_BORDER)

def _lo(**extra):
    """Merge chart-specific kwargs over the base _L dict (avoids duplicate-key errors)."""
    merged = dict(**_L)
    if 'legend' not in extra:
        merged['legend'] = _LEGEND
    merged.update(extra)
    return merged

# ── Pre-aggregated dataset statistics (58,645 records from train_loan.csv) ────
# All values computed offline — no CSV needed at runtime
STATS = dict(
    total=58645, approved=50295, rejected=8350,
    approval_rate=85.76, default_rate=14.24,
    avg_amount=9217.6, avg_rate=10.68,
    avg_income=64046, avg_dti=0.159, avg_age=27.6,
)

# Grade: approval rates
GRADE_DATA = dict(
    grades=['A','B','C','D','E','F','G'],
    rates=[95.1, 89.8, 86.5, 40.6, 37.5, 38.9, 18.2],
    totals=[20984, 20400, 11036, 5034, 1009, 149, 33],
)

# Loan purpose: approval rates
INTENT_DATA = dict(
    intents=['VENTURE','EDUCATION','PERSONAL','HOMEIMPROVEMENT','MEDICAL','DEBTCONSOLIDATION'],
    labels=['Venture / 벤처','Education / 교육','Personal / 개인','Home Improve / 주택개선',
            'Medical / 의료','Debt Consol. / 부채통합'],
    rates=[90.7, 89.2, 86.7, 82.6, 82.2, 81.1],
)

# Home ownership: approval rates
OWN_DATA = dict(
    cats=['OWN','MORTGAGE','OTHER','RENT'],
    rates=[98.6, 94.0, 83.1, 77.7],
    colors=[C_GREEN, C_TEAL, C_PURPLE, C_GOLD],
)

# Default history: approval rates
DEFAULT_DATA = dict(
    cats=['No Default History\n신용 불량 이력 없음', 'Prior Default\n신용 불량 이력 있음'],
    rates=[88.5, 70.1],
    colors=[C_GREEN, C_RED],
)

# Income histogram (binned, 99th pct capped)
INCOME_BINS = ['<20k','20-30k','30-40k','40-50k','50-60k','60-70k','70-80k','80-100k','>100k']
INCOME_APPROVED = [272, 3494, 6397, 8552, 8866, 5848, 5224, 6047, 5595]
INCOME_REJECTED = [256, 1642, 1865, 1452, 1327,  697,  545,  345,  221]

# Loan amount histogram (binned)
AMOUNT_BINS = ['<2k','2-4k','4-6k','6-8k','8-10k','10-12k','12-15k','15-20k','20-25k','>25k']
AMOUNT_APPROVED = [2117, 6759, 11265, 7369, 8518, 4775, 4684, 3100, 1465, 243]
AMOUNT_REJECTED = [ 260,  832,  1185,  907, 1399,  789, 1221, 1080,  568, 109]

# Box plot pre-computed stats: [q1, median, q3, mean, lower_fence, upper_fence]
DTI_APPROVED = [0.08,  0.13, 0.20, 0.145,  0.0,  0.40]
DTI_REJECTED = [0.14,  0.25, 0.33, 0.244,  0.0,  0.60]
RATE_APPROVED = [7.51, 10.39, 12.42, 10.26,  5.42, 18.0]
RATE_REJECTED = [11.11, 13.79, 15.58, 13.20,  7.0, 23.22]

# ── EDA chart builders (fast — from hardcoded arrays, no file I/O) ─────────────
def _chart_donut():
    fig = go.Figure(go.Pie(
        labels=['✅ Approved / 승인', '❌ Default / 부도'],
        values=[STATS['approved'], STATS['rejected']],
        hole=0.62, marker_colors=[C_GREEN, C_RED],
        textinfo='percent+label', textfont_size=12,
        hovertemplate='%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>',
    ))
    fig.add_annotation(
        text=f"<b>{STATS['total']:,}</b><br><span style='font-size:11px'>records</span>",
        x=0.5, y=0.5, font=dict(size=14, color=C_TEXT), showarrow=False,
    )
    fig.update_layout(**_lo(
        title='Loan Outcome Distribution<br><sub>대출 결과 분포</sub>',
        legend=dict(orientation='h', y=-0.1),
    ))
    return fig


def _chart_grade():
    colors = [C_GREEN if r > 70 else C_GOLD if r > 45 else C_RED
              for r in GRADE_DATA['rates']]
    fig = go.Figure(go.Bar(
        x=GRADE_DATA['grades'], y=GRADE_DATA['rates'],
        marker_color=colors,
        text=[f"{r}%" for r in GRADE_DATA['rates']], textposition='outside',
        customdata=GRADE_DATA['totals'],
        hovertemplate='Grade %{x}<br>Approval Rate: %{y}%<br>Total: %{customdata:,}<extra></extra>',
    ))
    fig.update_layout(**_lo(
        title='Approval Rate by Loan Grade<br><sub>신용 등급별 승인율</sub>',
        xaxis_title='Loan Grade / 신용 등급',
        yaxis=dict(title='Approval Rate (%) / 승인율', range=[0, 110], gridcolor=C_BORDER),
    ))
    return fig


def _chart_intent():
    fig = go.Figure(go.Bar(
        x=INTENT_DATA['rates'], y=INTENT_DATA['labels'],
        orientation='h', marker_color=C_BLUE,
        text=[f"{r}%" for r in INTENT_DATA['rates']], textposition='outside',
        hovertemplate='%{y}<br>Approval Rate: %{x}%<extra></extra>',
    ))
    fig.update_layout(**_lo(
        title='Approval Rate by Loan Purpose<br><sub>대출 목적별 승인율</sub>',
        xaxis=dict(title='Approval Rate (%) / 승인율', range=[0, 100], gridcolor=C_BORDER),
    ))
    return fig


def _chart_ownership():
    fig = go.Figure(go.Bar(
        x=OWN_DATA['cats'], y=OWN_DATA['rates'],
        marker_color=OWN_DATA['colors'],
        text=[f"{r}%" for r in OWN_DATA['rates']], textposition='outside',
        hovertemplate='%{x}<br>Approval Rate: %{y}%<extra></extra>',
    ))
    fig.update_layout(**_lo(
        title='Approval Rate by Home Ownership<br><sub>주택 소유 형태별 승인율</sub>',
        yaxis=dict(title='Approval Rate (%) / 승인율', range=[0, 110], gridcolor=C_BORDER),
    ))
    return fig


def _chart_income_hist():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=INCOME_BINS, y=INCOME_APPROVED, name='✅ Approved',
                         marker_color=C_GREEN, opacity=0.8))
    fig.add_trace(go.Bar(x=INCOME_BINS, y=INCOME_REJECTED, name='❌ Default',
                         marker_color=C_RED, opacity=0.8))
    fig.update_layout(**_lo(
        barmode='group',
        title='Annual Income Distribution by Outcome<br><sub>결과별 연간 소득 분포</sub>',
        xaxis_title='Income Range / 소득 구간',
        yaxis=dict(title='Count / 건수', gridcolor=C_BORDER),
        legend=dict(orientation='h', y=1.1),
    ))
    return fig


def _chart_amount_hist():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=AMOUNT_BINS, y=AMOUNT_APPROVED, name='✅ Approved',
                         marker_color=C_GREEN, opacity=0.8))
    fig.add_trace(go.Bar(x=AMOUNT_BINS, y=AMOUNT_REJECTED, name='❌ Default',
                         marker_color=C_RED, opacity=0.8))
    fig.update_layout(**_lo(
        barmode='group',
        title='Loan Amount Distribution by Outcome<br><sub>결과별 대출 금액 분포</sub>',
        xaxis_title='Loan Amount / 대출 금액',
        yaxis=dict(title='Count / 건수', gridcolor=C_BORDER),
        legend=dict(orientation='h', y=1.1),
    ))
    return fig


def _box_trace(name, color, stats):
    q1, med, q3, mean, lo, hi = stats
    return go.Box(
        name=name, marker_color=color, line_color=color,
        q1=[q1], median=[med], q3=[q3], mean=[mean],
        lowerfence=[lo], upperfence=[hi],
        boxmean='sd',
        hovertemplate=f'{name}<br>Q1: {q1}<br>Median: {med}<br>Q3: {q3}<br>Mean: {mean}<extra></extra>',
    )


def _chart_dti():
    fig = go.Figure([
        _box_trace('✅ Approved / 승인', C_GREEN, DTI_APPROVED),
        _box_trace('❌ Default / 부도',  C_RED,   DTI_REJECTED),
    ])
    fig.update_layout(**_lo(
        title='DTI Ratio by Outcome<br><sub>결과별 부채 소득 비율</sub>',
        yaxis=dict(title='DTI Ratio / 부채 소득 비율', gridcolor=C_BORDER),
    ))
    return fig


def _chart_rate():
    fig = go.Figure([
        _box_trace('✅ Approved / 승인', C_GREEN, RATE_APPROVED),
        _box_trace('❌ Default / 부도',  C_RED,   RATE_REJECTED),
    ])
    fig.update_layout(**_lo(
        title='Interest Rate by Outcome<br><sub>결과별 금리</sub>',
        yaxis=dict(title='Interest Rate (%) / 금리', gridcolor=C_BORDER),
    ))
    return fig


def _chart_default_history():
    fig = go.Figure(go.Bar(
        x=DEFAULT_DATA['cats'], y=DEFAULT_DATA['rates'],
        marker_color=DEFAULT_DATA['colors'],
        text=[f"{r}%" for r in DEFAULT_DATA['rates']], textposition='outside',
        hovertemplate='%{x}<br>Approval Rate: %{y}%<extra></extra>',
    ))
    fig.update_layout(**_lo(
        title='Approval Rate by Default History<br><sub>신용 불량 이력별 승인율</sub>',
        yaxis=dict(title='Approval Rate (%) / 승인율', range=[0, 105], gridcolor=C_BORDER),
    ))
    return fig


# Pre-compute all charts at startup — fast because data is hardcoded arrays
FIG_DONUT   = _chart_donut()
FIG_GRADE   = _chart_grade()
FIG_INTENT  = _chart_intent()
FIG_OWN     = _chart_ownership()
FIG_INCOME  = _chart_income_hist()
FIG_AMOUNT  = _chart_amount_hist()
FIG_DTI     = _chart_dti()
FIG_RATE    = _chart_rate()
FIG_DEFAULT = _chart_default_history()

# ── Feature labels (English / 한국어) ─────────────────────────────────────────
_FL = {
    'num__person_age':           'Age / 나이',
    'num__person_income':        'Annual Income / 연간 소득',
    'num__person_emp_length':    'Employment Length / 고용 기간',
    'num__loan_amnt':            'Loan Amount / 대출 금액',
    'num__loan_int_rate':        'Interest Rate / 금리',
    'num__loan_percent_income':  'DTI Ratio / 부채 소득 비율',
    'cat__person_home_ownership_RENT':     'Home Ownership: Rent / 임대',
    'cat__person_home_ownership_OWN':      'Home Ownership: Own / 자가',
    'cat__person_home_ownership_MORTGAGE': 'Home Ownership: Mortgage / 담보',
    'cat__person_home_ownership_OTHER':    'Home Ownership: Other / 기타',
    'cat__loan_intent_PERSONAL':           'Loan Purpose: Personal / 개인',
    'cat__loan_intent_EDUCATION':          'Loan Purpose: Education / 교육',
    'cat__loan_intent_MEDICAL':            'Loan Purpose: Medical / 의료',
    'cat__loan_intent_VENTURE':            'Loan Purpose: Venture / 벤처',
    'cat__loan_intent_HOMEIMPROVEMENT':    'Loan Purpose: Home Improvement / 주택개선',
    'cat__loan_intent_DEBTCONSOLIDATION':  'Loan Purpose: Debt Consolidation / 부채통합',
    'cat__loan_grade_A': 'Loan Grade A (Best) / A등급',
    'cat__loan_grade_B': 'Loan Grade B / B등급',
    'cat__loan_grade_C': 'Loan Grade C / C등급',
    'cat__loan_grade_D': 'Loan Grade D / D등급',
    'cat__loan_grade_E': 'Loan Grade E / E등급',
    'cat__loan_grade_F': 'Loan Grade F / F등급',
    'cat__loan_grade_G': 'Loan Grade G (Worst) / G등급',
    'cat__cb_person_default_on_file_N': 'No Prior Default / 불량 이력 없음',
    'cat__cb_person_default_on_file_Y': 'Prior Default on File / 불량 이력 있음',
}
_ADVICE = [
    ('loan_percent_income', 'Reduce DTI: borrow less or increase income / DTI 개선: 대출 금액 줄이기 또는 소득 증가'),
    ('loan_int_rate',       'Improve credit score to qualify for lower rates / 낮은 금리를 위해 신용 점수 개선'),
    ('loan_amnt',           'Consider requesting a smaller loan amount / 더 적은 대출 금액 요청 고려'),
    ('person_income',       'Add secondary income sources before applying / 추가 소득원 확보 후 재신청'),
    ('person_emp_length',   'Maintain stable employment to increase tenure / 안정적인 고용 유지로 근무 기간 연장'),
    ('loan_grade_D',        'Improve credit score to reach Grade A/B/C / 신용 점수 개선으로 A/B/C 등급 달성'),
    ('loan_grade_E',        'Improve credit score to reach Grade A/B/C / 신용 점수 개선으로 A/B/C 등급 달성'),
    ('loan_grade_F',        'Improve credit score to reach Grade A/B/C / 신용 점수 개선으로 A/B/C 등급 달성'),
    ('loan_grade_G',        'Improve credit score to reach Grade A/B/C / 신용 점수 개선으로 A/B/C 등급 달성'),
    ('default_on_file_Y',   'Rebuild credit history with secured credit card / 보증 신용카드로 신용 이력 재건'),
    ('home_ownership_RENT', 'Owning or mortgaging property improves credit profile / 주택 소유/담보가 신용 프로필 향상'),
]


def _label(raw):
    return _FL.get(raw, raw.replace('num__','').replace('cat__','').replace('_',' ').title())


def _advice(name):
    for key, text in _ADVICE:
        if key in name:
            return text
    return None


# ── SHAP explanation builder ──────────────────────────────────────────────────
def _build_explanation(sv, feature_names, decision, prob_approve):
    prob_default = 1 - prob_approve
    pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)
    pos = [(n, v) for n, v in pairs if v > 0]
    neg = [(n, v) for n, v in pairs if v < 0]
    approved = '✅' in decision

    lines = []
    if approved:
        lines.append(f"## ✅ Application APPROVED / 승인\n")
        lines.append(f"**Approval Probability / 승인 확률:** {prob_approve:.1%} &nbsp;·&nbsp; "
                     f"**Default Risk / 부도 위험:** {prob_default:.1%}\n\n---\n")
        lines.append("### Why Approved? / 승인 이유\n")
        lines.append("The following factors **support** this application *(blue bars in SHAP chart / SHAP 차트의 파란색 막대)*:\n\n")
        for name, val in pos[:4]:
            strength = ("very strong / 매우 강함" if abs(val) > 0.5 else
                        "strong / 강함" if abs(val) > 0.25 else
                        "moderate / 보통" if abs(val) > 0.1 else "slight / 약함")
            lines.append(f"- **{_label(name)}** — positive impact, {strength} `(+{val:.3f})`\n")
        if neg:
            lines.append("\n---\n### ⚠️ Risk Factors Present / 잠재 위험 요소\n")
            lines.append("These factors are present but **not enough to reject** *(red bars)*:\n\n")
            for name, val in neg[:3]:
                lines.append(f"- **{_label(name)}** `({val:.3f})`\n")
            lines.append("\n💡 Improving these factors will further strengthen your credit profile.\n"
                         "이러한 요소를 개선하면 신용 프로필이 더욱 강화됩니다.\n")
    else:
        lines.append(f"## ❌ Application REJECTED / 거절\n")
        lines.append(f"**Default Risk / 부도 위험:** {prob_default:.1%} &nbsp;·&nbsp; "
                     f"**Approval Probability / 승인 확률:** {prob_approve:.1%}\n\n---\n")
        lines.append("### Why Rejected? / 거절 이유\n")
        lines.append("The following factors **pulled the application down** *(red bars in SHAP chart / SHAP 차트의 빨간색 막대)*:\n\n")
        for name, val in neg[:4]:
            severity = ("very high / 매우 높음" if abs(val) > 0.5 else
                        "high / 높음" if abs(val) > 0.25 else "moderate / 보통")
            lines.append(f"- **{_label(name)}** — risk {severity} `({val:.3f})`\n")
        if pos:
            lines.append("\n---\n### ✅ Positive Factors / 긍정적 요소\n")
            lines.append("These factors help but are **not enough to offset the risk**:\n\n")
            for name, val in pos[:3]:
                lines.append(f"- **{_label(name)}** `(+{val:.3f})`\n")
        lines.append("\n---\n### 💡 How to Improve / 개선 방법\n\n")
        advices = [a for a in (_advice(n) for n, _ in neg[:4]) if a]
        for a in (advices or ["Contact a credit counselor for personalized advice. / 개인 맞춤 상담을 위해 신용 상담사에게 문의하세요."]):
            lines.append(f"- {a}\n")

    return "".join(lines)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_loan(age, income, home_ownership, emp_length, intent,
                 grade, amount, rate, percent_income, default_hist):
    raw = pd.DataFrame(
        [[age, income, home_ownership, emp_length, intent,
          grade, amount, rate, percent_income, default_hist]],
        columns=['person_age','person_income','person_home_ownership',
                 'person_emp_length','loan_intent','loan_grade',
                 'loan_amnt','loan_int_rate','loan_percent_income',
                 'cb_person_default_on_file']
    )
    processed     = preprocessor.transform(raw)
    feature_names = preprocessor.get_feature_names_out()
    input_df      = pd.DataFrame(processed, columns=feature_names)

    # class 1 = fully paid (approved), class 0 = default
    prob_approve = float(model.predict_proba(input_df)[0][1])
    prob_default = 1.0 - prob_approve
    decision = "✅  APPROVED / 승인" if prob_approve > 0.5 else "❌  REJECTED / 거절"

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    base_val    = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1])
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    fig = plt.figure(figsize=(12, 7))
    shap.plots.waterfall(
        shap.Explanation(values=sv, base_values=base_val,
                         data=input_df.iloc[0], feature_names=list(feature_names)),
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_plot.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    explanation = _build_explanation(sv, list(feature_names), decision, prob_approve)
    return decision, f"{prob_default:.1%}", f"{prob_approve:.1%}", "shap_plot.png", explanation


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
footer { display: none !important; }
.kpi-grid {
    display: grid; grid-template-columns: repeat(6, 1fr);
    gap: 10px; margin: 14px 0 8px 0;
}
.kpi-card {
    background: #1c2333; border: 1px solid #30363d;
    border-radius: 10px; padding: 18px 10px; text-align: center;
}
.kpi-val {
    font-size: 1.65rem; font-weight: 800; color: #e8b84b; margin: 0; line-height: 1.15;
}
.kpi-lbl {
    font-size: 0.72rem; color: #9ca3af; margin-top: 5px;
    text-transform: uppercase; letter-spacing: 0.05em; line-height: 1.4;
}
.insight {
    background: #1c2333; border-left: 4px solid #3b82f6;
    border-radius: 6px; padding: 12px 18px; margin: 6px 0;
    color: #e5e7eb; font-size: 0.9rem; line-height: 1.6;
}
#decision-box textarea {
    text-align: center !important; font-size: 1.35rem !important;
    font-weight: 700 !important; padding: 18px !important;
}
#submit-btn { font-size: 1.05rem !important; font-weight: 600 !important; }
"""

KPI_HTML = f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <p class="kpi-val">{STATS['total']:,}</p>
        <p class="kpi-lbl">Total Records<br>총 데이터</p>
    </div>
    <div class="kpi-card">
        <p class="kpi-val" style="color:#10b981;">{STATS['approval_rate']:.1f}%</p>
        <p class="kpi-lbl">Approval Rate<br>승인율</p>
    </div>
    <div class="kpi-card">
        <p class="kpi-val" style="color:#ef4444;">{STATS['default_rate']:.1f}%</p>
        <p class="kpi-lbl">Default Rate<br>부도율</p>
    </div>
    <div class="kpi-card">
        <p class="kpi-val">${STATS['avg_amount']:,.0f}</p>
        <p class="kpi-lbl">Avg. Loan Amount<br>평균 대출 금액</p>
    </div>
    <div class="kpi-card">
        <p class="kpi-val">{STATS['avg_rate']:.1f}%</p>
        <p class="kpi-lbl">Avg. Interest Rate<br>평균 금리</p>
    </div>
    <div class="kpi-card">
        <p class="kpi-val">{STATS['avg_dti']:.3f}</p>
        <p class="kpi-lbl">Avg. DTI Ratio<br>평균 DTI</p>
    </div>
</div>
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue",
                         font=gr.themes.GoogleFont("Inter")),
    title="AI Loan Approval System",
    css=CSS,
) as iface:

    gr.HTML("""
        <div style="background:linear-gradient(135deg,#1e1b4b 0%,#0d1117 100%);
                    border-radius:12px; padding:28px 32px; margin-bottom:12px;">
            <h1 style="font-size:1.9rem;font-weight:800;margin:0;color:#fff;">
                🏦 AI Loan Approval System
            </h1>
            <p style="color:#9ca3af;font-size:0.92rem;margin:8px 0 0 0;">
                XGBoost · SHAP Explainability · EDA Dashboard &nbsp;|&nbsp;
                <span style="color:#e8b84b;">58,645 records · 85.76% approval rate</span>
            </p>
        </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — Data Overview
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📊 Data Overview / 데이터 개요"):

            gr.HTML(KPI_HTML)

            # Row 1
            with gr.Row():
                gr.Plot(value=FIG_DONUT)
                gr.Plot(value=FIG_GRADE)

            gr.HTML("""
            <div class="insight">
                💡 <b>Insight:</b> Grades <b>A and B</b> achieve >89% approval rate.
                Grades <b>E, F, G</b> drop below 40% — loan grade is the strongest single predictor of approval.
                &nbsp;|&nbsp; <b>인사이트:</b> A·B 등급은 89% 이상의 승인율을 보이며, E·F·G 등급은 40% 미만으로 급락합니다.
                신용 등급이 승인 여부를 가장 강하게 예측하는 변수입니다.
            </div>""")

            # Row 2
            with gr.Row():
                gr.Plot(value=FIG_INTENT)
                gr.Plot(value=FIG_OWN)

            gr.HTML("""
            <div class="insight">
                💡 <b>Insight:</b> <b>Venture</b> and <b>Education</b> loans have the highest approval rates (>89%).
                Homeowners have a 98.6% approval rate vs 77.7% for renters — home ownership signals financial stability.
                &nbsp;|&nbsp; <b>인사이트:</b> 벤처·교육 목적 대출의 승인율이 가장 높습니다. 자가 소유자의 승인율(98.6%)은
                임대 거주자(77.7%)보다 훨씬 높아, 주택 소유가 재정적 안정성의 신호로 작용합니다.
            </div>""")

            # Row 3
            with gr.Row():
                gr.Plot(value=FIG_INCOME)
                gr.Plot(value=FIG_AMOUNT)

            gr.HTML("""
            <div class="insight">
                💡 <b>Insight:</b> Approved applicants have a higher income concentration in the $40k–$80k range.
                Defaulted applicants borrow more — higher loan amounts relative to income increase default risk.
                &nbsp;|&nbsp; <b>인사이트:</b> 승인된 신청자는 4만~8만 달러 소득 구간에 집중됩니다.
                부도 그룹은 더 큰 금액을 빌리는 경향이 있어, 소득 대비 대출 비율이 부도 위험을 높입니다.
            </div>""")

            # Row 4
            with gr.Row():
                gr.Plot(value=FIG_DTI)
                gr.Plot(value=FIG_RATE)

            gr.HTML("""
            <div class="insight">
                💡 <b>Insight:</b> Defaulted loans have a <b>median DTI of 0.25</b> vs <b>0.13</b> for approved loans.
                Interest rates are also higher for defaults (13.8% median vs 10.4%).
                DTI &gt; 0.4 is a critical risk threshold.
                &nbsp;|&nbsp; <b>인사이트:</b> 부도 대출의 중앙값 DTI는 0.25로, 승인 대출(0.13)보다 훨씬 높습니다.
                금리도 부도 그룹이 더 높습니다(중앙값 13.8% vs 10.4%). DTI > 0.4는 고위험 기준점입니다.
            </div>""")

            # Row 5
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Plot(value=FIG_DEFAULT)
                with gr.Column(scale=1):
                    gr.HTML(f"""
                    <div style="padding:16px 20px;">
                        <h3 style="color:{C_TEXT};margin-bottom:14px;font-size:1rem;font-weight:700;">
                            📌 Key Insights Summary / 핵심 인사이트 요약
                        </h3>
                        <div class="insight" style="border-color:{C_GREEN};margin-bottom:8px;">
                            ✅ <b>Strong Profile / 우수 프로필:</b><br>
                            Grade A/B · High income · Low DTI · Homeowner · No default history<br>
                            <span style="color:{C_MUTED};">A/B등급 · 높은 소득 · 낮은 DTI · 자가 소유 · 불량 이력 없음</span>
                        </div>
                        <div class="insight" style="border-color:{C_RED};margin-bottom:8px;">
                            ❌ <b>High Risk Profile / 고위험 프로필:</b><br>
                            Grade E–G · DTI &gt; 0.4 · Rate &gt; 15% · Renter · Prior default<br>
                            <span style="color:{C_MUTED};">E–G등급 · DTI &gt; 0.4 · 금리 &gt; 15% · 임차인 · 불량 이력 있음</span>
                        </div>
                        <div class="insight" style="border-color:{C_GOLD};">
                            💡 <b>Top Predictors / 주요 예측 변수:</b><br>
                            Loan Grade &gt; DTI Ratio &gt; Interest Rate &gt; Default History<br>
                            <span style="color:{C_MUTED};">신용 등급 &gt; DTI 비율 &gt; 금리 &gt; 불량 이력</span>
                        </div>
                    </div>""")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — Loan Predictor
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🤖 Loan Predictor / 대출 예측기"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Applicant Information / 신청자 정보")
                    age = gr.Number(label="Age / 나이 (years)", value=26,
                                    minimum=18, maximum=100)
                    income = gr.Number(label="Annual Income / 연간 소득 ($)", value=44000)
                    emp_length = gr.Number(label="Employment Length / 고용 기간 (years)",
                                           value=6, minimum=0, maximum=50)
                    home_ownership = gr.Dropdown(
                        choices=["RENT", "OWN", "MORTGAGE", "OTHER"],
                        label="Home Ownership / 주택 소유 형태", value="RENT")
                    default_history = gr.Radio(
                        choices=["N", "Y"],
                        label="Prior Default on File? / 신용 불량 이력?", value="N")

                with gr.Column(scale=1):
                    gr.Markdown("### 💳 Loan Details / 대출 세부 정보")
                    amount = gr.Number(label="Loan Amount / 대출 금액 ($)", value=11000)
                    rate = gr.Number(label="Interest Rate / 금리 (%)", value=10.37)
                    percent_income = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        label="DTI Ratio / 부채 소득 비율 (0.26 = 26%)", value=0.26)
                    intent = gr.Dropdown(
                        choices=["PERSONAL","EDUCATION","MEDICAL",
                                 "VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"],
                        label="Loan Purpose / 대출 목적", value="PERSONAL")
                    grade = gr.Dropdown(
                        choices=["A","B","C","D","E","F","G"],
                        label="Loan Grade / 신용 등급 (A=Best / 최고, G=Worst / 최저)", value="B")

            submit_btn = gr.Button(
                "🔍  Analyze Application / 신청서 분석",
                variant="primary", size="lg", elem_id="submit-btn")

            gr.HTML(f'<hr style="border-color:{C_BORDER};margin:16px 0;"/>')
            gr.Markdown("### 📊 Decision Result / 판정 결과")

            with gr.Row():
                decision_out = gr.Textbox(
                    label="Final Decision / 최종 결정",
                    interactive=False, elem_id="decision-box", scale=2)
                with gr.Column(scale=1):
                    prob_default_out = gr.Textbox(
                        label="🔴 Default Risk / 부도 위험", interactive=False)
                    prob_approve_out = gr.Textbox(
                        label="🟢 Approval Probability / 승인 확률", interactive=False)

            gr.HTML(f'<hr style="border-color:{C_BORDER};margin:16px 0;"/>')

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 SHAP Analysis / SHAP 분석")
                    gr.Markdown(
                        "> 🔵 **Blue bars / 파란 막대** → push toward **Approval / 승인** (lower risk)  \n"
                        "> 🔴 **Red bars / 빨간 막대** → push toward **Rejection / 거절** (higher risk)  \n"
                        "> Longer bar = stronger influence on the decision / 막대가 길수록 결정에 더 큰 영향"
                    )
                    shap_out = gr.Image(label="SHAP Waterfall Plot")

                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Detailed Explanation / 상세 설명")
                    explanation_out = gr.Markdown(
                        value="> *Click **Analyze Application** to see the full explanation.*  \n"
                              "> *신청서 분석 버튼을 눌러 상세 설명을 확인하세요.*"
                    )

            gr.HTML(f'<hr style="border-color:{C_BORDER};margin:16px 0;"/>')
            gr.Markdown("### 📋 Sample Applications / 샘플 신청서 — Click to load / 클릭하여 불러오기")
            gr.Examples(
                examples=[
                    [32, 60000, "MORTGAGE", 12, "VENTURE",  "A", 5000,  7.50, 0.04, "N"],
                    [26, 44000, "RENT",      6, "PERSONAL", "B", 11000, 10.37, 0.26, "N"],
                    [22, 25000, "RENT",      1, "PERSONAL", "E", 15000, 18.50, 0.60, "Y"],
                ],
                inputs=[age, income, home_ownership, emp_length, intent,
                        grade, amount, rate, percent_income, default_history],
                label="✅ Low Risk (Grade A)  |  ⚠️ Borderline (Grade B)  |  ❌ High Risk (Grade E+Default)"
            )

            submit_btn.click(
                fn=predict_loan,
                inputs=[age, income, home_ownership, emp_length, intent,
                        grade, amount, rate, percent_income, default_history],
                outputs=[decision_out, prob_default_out, prob_approve_out,
                         shap_out, explanation_out]
            )

if __name__ == "__main__":
    iface.launch()
