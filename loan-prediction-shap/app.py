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

# ── Load model & preprocessor (lightweight — no heavy computation here) ────────
model = xgb.XGBClassifier()
if os.path.exists('xgb_loan_model.json'):
    model.load_model('xgb_loan_model.json')

preprocessor = joblib.load('preprocessor.pkl') if os.path.exists('preprocessor.pkl') else None

# CSV path: HF Space (same dir) or local dev (parent dir)
_csv = 'train_loan.csv' if os.path.exists('train_loan.csv') else \
       '../train_loan.csv' if os.path.exists('../train_loan.csv') else None

# ── Design tokens ─────────────────────────────────────────────────────────────
C_GREEN  = '#10b981'
C_RED    = '#ef4444'
C_BLUE   = '#3b82f6'
C_GOLD   = '#e8b84b'
C_PURPLE = '#8b5cf6'
C_TEAL   = '#14b8a6'
C_BG     = '#161b22'
C_BORDER = '#30363d'
C_TEXT   = '#e5e7eb'
C_MUTED  = '#9ca3af'

_L = dict(  # shared Plotly layout
    paper_bgcolor=C_BG, plot_bgcolor=C_BG,
    font=dict(color=C_TEXT, family='Inter, sans-serif'),
    margin=dict(l=48, r=32, t=52, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=C_BORDER),
    xaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER),
    yaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER),
)

# ── Feature label map ─────────────────────────────────────────────────────────
_FL = {
    'num__person_age':           'Tuổi người vay',
    'num__person_income':        'Thu nhập hàng năm',
    'num__person_emp_length':    'Thâm niên công tác',
    'num__loan_amnt':            'Số tiền vay',
    'num__loan_int_rate':        'Lãi suất khoản vay',
    'num__loan_percent_income':  'Tỷ lệ nợ/thu nhập (DTI)',
    'cat__person_home_ownership_RENT':     'Nhà thuê (RENT)',
    'cat__person_home_ownership_OWN':      'Nhà sở hữu (OWN)',
    'cat__person_home_ownership_MORTGAGE': 'Nhà thế chấp (MORTGAGE)',
    'cat__person_home_ownership_OTHER':    'Nhà ở khác',
    'cat__loan_intent_PERSONAL':           'Mục đích: Cá nhân',
    'cat__loan_intent_EDUCATION':          'Mục đích: Giáo dục',
    'cat__loan_intent_MEDICAL':            'Mục đích: Y tế',
    'cat__loan_intent_VENTURE':            'Mục đích: Kinh doanh',
    'cat__loan_intent_HOMEIMPROVEMENT':    'Mục đích: Cải tạo nhà',
    'cat__loan_intent_DEBTCONSOLIDATION':  'Mục đích: Trả nợ',
    'cat__loan_grade_A': 'Hạng tín dụng A (Tốt nhất)',
    'cat__loan_grade_B': 'Hạng tín dụng B',
    'cat__loan_grade_C': 'Hạng tín dụng C',
    'cat__loan_grade_D': 'Hạng tín dụng D',
    'cat__loan_grade_E': 'Hạng tín dụng E',
    'cat__loan_grade_F': 'Hạng tín dụng F',
    'cat__loan_grade_G': 'Hạng tín dụng G (Tệ nhất)',
    'cat__cb_person_default_on_file_N': 'Không có lịch sử nợ xấu',
    'cat__cb_person_default_on_file_Y': 'Có lịch sử nợ xấu',
}
_ADVICE = [
    ('loan_percent_income', 'Giảm DTI: cân nhắc vay số tiền nhỏ hơn hoặc bổ sung nguồn thu nhập'),
    ('loan_int_rate',       'Cải thiện điểm tín dụng để được hưởng lãi suất thấp hơn'),
    ('loan_amnt',           'Giảm số tiền vay xuống mức phù hợp hơn với thu nhập'),
    ('person_income',       'Bổ sung thu nhập phụ hoặc chờ tăng lương trước khi vay'),
    ('person_emp_length',   'Duy trì việc làm ổn định để tăng thâm niên công tác'),
    ('loan_grade_D',        'Cải thiện điểm tín dụng để đạt hạng A, B hoặc C'),
    ('loan_grade_E',        'Cải thiện điểm tín dụng để đạt hạng A, B hoặc C'),
    ('loan_grade_F',        'Cải thiện điểm tín dụng để đạt hạng A, B hoặc C'),
    ('loan_grade_G',        'Cải thiện điểm tín dụng để đạt hạng A, B hoặc C'),
    ('default_on_file_Y',   'Xây dựng lại lịch sử tín dụng qua thẻ tín dụng có bảo đảm'),
    ('home_ownership_RENT', 'Chuyển sang sở hữu/thế chấp nhà sẽ tăng uy tín tín dụng'),
]


def _label(raw):
    return _FL.get(raw, raw.replace('num__', '').replace('cat__', '').replace('_', ' ').title())


def _advice(name):
    for key, text in _ADVICE:
        if key in name:
            return text
    return None


# ── EDA chart builders (called lazily, not at import time) ────────────────────
def _make_donut(df):
    counts = df['loan_status'].value_counts()
    fig = go.Figure(go.Pie(
        labels=['✅ Approved', '❌ Default'],
        values=[counts.get(0, 0), counts.get(1, 0)],
        hole=0.62,
        marker_colors=[C_GREEN, C_RED],
        textinfo='percent+label', textfont_size=13,
    ))
    fig.add_annotation(
        text=f"<b>{len(df):,}</b><br><span style='font-size:11px'>hồ sơ</span>",
        x=0.5, y=0.5, font=dict(size=15, color=C_TEXT), showarrow=False
    )
    fig.update_layout(title='Phân Bố Kết Quả Khoản Vay',
                      legend=dict(orientation='h', y=-0.08), **_L)
    return fig


def _make_grade(df):
    g = df.groupby('loan_grade').agg(
        total=('loan_status', 'count'),
        approved=('loan_status', lambda x: (x == 0).sum())
    ).reset_index()
    g['rate'] = g['approved'] / g['total'] * 100
    g = g.sort_values('loan_grade')
    colors = [C_GREEN if r > 70 else C_GOLD if r > 45 else C_RED for r in g['rate']]
    fig = go.Figure(go.Bar(
        x=g['loan_grade'], y=g['rate'], marker_color=colors,
        text=[f"{r:.1f}%" for r in g['rate']], textposition='outside',
        hovertemplate='Grade %{x}: %{y:.1f}%<extra></extra>',
    ))
    fig.update_layout(title='Tỷ Lệ Duyệt Theo Hạng Tín Dụng',
                      xaxis_title='Loan Grade',
                      yaxis=dict(title='Tỷ lệ duyệt (%)', range=[0, 108], gridcolor=C_BORDER),
                      **_L)
    return fig


def _make_intent(df):
    LABELS = {'PERSONAL': 'Cá nhân', 'EDUCATION': 'Giáo dục', 'MEDICAL': 'Y tế',
              'VENTURE': 'Kinh doanh', 'HOMEIMPROVEMENT': 'Cải tạo nhà',
              'DEBTCONSOLIDATION': 'Trả nợ'}
    g = df.groupby('loan_intent').agg(
        total=('loan_status', 'count'),
        approved=('loan_status', lambda x: (x == 0).sum())
    ).reset_index()
    g['rate'] = g['approved'] / g['total'] * 100
    g['label'] = g['loan_intent'].map(LABELS)
    g = g.sort_values('rate', ascending=True)
    fig = go.Figure(go.Bar(
        x=g['rate'], y=g['label'], orientation='h', marker_color=C_BLUE,
        text=[f"{r:.1f}%" for r in g['rate']], textposition='outside',
    ))
    fig.update_layout(title='Tỷ Lệ Duyệt Theo Mục Đích Vay',
                      xaxis=dict(title='Tỷ lệ duyệt (%)', range=[0, 100], gridcolor=C_BORDER),
                      **_L)
    return fig


def _make_ownership(df):
    g = df.groupby('person_home_ownership').agg(
        total=('loan_status', 'count'),
        approved=('loan_status', lambda x: (x == 0).sum())
    ).reset_index()
    g['rate'] = g['approved'] / g['total'] * 100
    pal = {'OWN': C_GREEN, 'MORTGAGE': C_TEAL, 'RENT': C_GOLD, 'OTHER': C_PURPLE}
    fig = go.Figure(go.Bar(
        x=g['person_home_ownership'], y=g['rate'],
        marker_color=[pal.get(o, C_BLUE) for o in g['person_home_ownership']],
        text=[f"{r:.1f}%" for r in g['rate']], textposition='outside',
    ))
    fig.update_layout(title='Tỷ Lệ Duyệt Theo Hình Thức Nhà Ở',
                      yaxis=dict(title='Tỷ lệ duyệt (%)', range=[0, 105], gridcolor=C_BORDER),
                      **_L)
    return fig


def _make_hist(df, col, title, xlabel):
    cap = df[col].quantile(0.99)
    app = df.loc[df['loan_status'] == 0, col].clip(upper=cap)
    rej = df.loc[df['loan_status'] == 1, col].clip(upper=cap)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=app, name='✅ Approved', opacity=0.75,
                               marker_color=C_GREEN, nbinsx=50))
    fig.add_trace(go.Histogram(x=rej, name='❌ Default', opacity=0.75,
                               marker_color=C_RED, nbinsx=50))
    fig.update_layout(barmode='overlay', title=title,
                      xaxis_title=xlabel, yaxis_title='Số lượng',
                      legend=dict(orientation='h', y=1.08), **_L)
    return fig


def _make_box(df, col, title, ylabel):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df.loc[df['loan_status'] == 0, col],
                         name='✅ Approved', marker_color=C_GREEN, boxpoints='outliers'))
    fig.add_trace(go.Box(y=df.loc[df['loan_status'] == 1, col],
                         name='❌ Default',  marker_color=C_RED,   boxpoints='outliers'))
    fig.update_layout(title=title,
                      yaxis=dict(title=ylabel, gridcolor=C_BORDER), **_L)
    return fig


def _make_default_bar(df):
    g = df.groupby('cb_person_default_on_file').agg(
        total=('loan_status', 'count'),
        approved=('loan_status', lambda x: (x == 0).sum())
    ).reset_index()
    g['rate'] = g['approved'] / g['total'] * 100
    labels_map = {'N': 'Không nợ xấu (N)', 'Y': 'Có lịch sử nợ xấu (Y)'}
    fig = go.Figure(go.Bar(
        x=[labels_map.get(v, v) for v in g['cb_person_default_on_file']],
        y=g['rate'],
        marker_color=[C_GREEN if v == 'N' else C_RED
                      for v in g['cb_person_default_on_file']],
        text=[f"{r:.1f}%" for r in g['rate']], textposition='outside',
    ))
    fig.update_layout(title='Tỷ Lệ Duyệt Theo Lịch Sử Nợ Xấu',
                      yaxis=dict(title='Tỷ lệ duyệt (%)', range=[0, 105], gridcolor=C_BORDER),
                      **_L)
    return fig


# ── Lazy EDA loader — called by iface.load(), NOT at import time ──────────────
def load_eda():
    """Compute all EDA figures when user opens the page (not during build)."""
    if _csv is None:
        empty = go.Figure()
        empty.update_layout(title='Data not found', **_L)
        no_data = "⚠️ Không tìm thấy `train_loan.csv`."
        return [empty] * 9 + [no_data]

    df = pd.read_csv(_csv)
    # Sample to cap memory on constrained environments
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)

    n_total     = len(df)
    n_approved  = int((df['loan_status'] == 0).sum())
    apr_rate    = n_approved / n_total * 100
    avg_amount  = df['loan_amnt'].mean()
    avg_rate    = df['loan_int_rate'].mean()
    avg_income  = df['person_income'].mean()
    avg_dti     = df['loan_percent_income'].mean()

    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <p class="kpi-val">{n_total:,}</p><p class="kpi-lbl">Tổng hồ sơ</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-val" style="color:#10b981;">{apr_rate:.1f}%</p>
            <p class="kpi-lbl">Tỷ lệ duyệt</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-val" style="color:#ef4444;">{100 - apr_rate:.1f}%</p>
            <p class="kpi-lbl">Tỷ lệ nợ xấu</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-val">${avg_amount:,.0f}</p><p class="kpi-lbl">Số tiền vay TB</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-val">{avg_rate:.1f}%</p><p class="kpi-lbl">Lãi suất TB</p>
        </div>
        <div class="kpi-card">
            <p class="kpi-val">{avg_dti:.2f}</p><p class="kpi-lbl">DTI ratio TB</p>
        </div>
    </div>
    """

    return [
        _make_donut(df),
        _make_grade(df),
        _make_intent(df),
        _make_ownership(df),
        _make_hist(df, 'person_income', 'Phân Bố Thu Nhập Theo Kết Quả', 'Thu nhập ($)'),
        _make_hist(df, 'loan_amnt',     'Phân Bố Số Tiền Vay Theo Kết Quả', 'Số tiền ($)'),
        _make_box(df, 'loan_percent_income', 'DTI Ratio Theo Kết Quả', 'DTI Ratio'),
        _make_box(df, 'loan_int_rate',       'Lãi Suất Theo Kết Quả',  'Lãi suất (%)'),
        _make_default_bar(df),
        kpi_html,
    ]


# ── SHAP explanation generator ────────────────────────────────────────────────
def _build_explanation(sv, feature_names, decision, prob_approve):
    prob_default = 1 - prob_approve
    pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)
    pos = [(n, v) for n, v in pairs if v > 0]
    neg = [(n, v) for n, v in pairs if v < 0]

    lines = []
    approved = '✅' in decision

    if approved:
        lines.append(f"## ✅ Khoản Vay Được DUYỆT\n")
        lines.append(f"Xác suất phê duyệt: **{prob_approve:.1%}** · Xác suất nợ xấu: **{prob_default:.1%}**\n\n---\n")
        lines.append("### Tại sao được duyệt?\n")
        lines.append("Các yếu tố dưới đây đang **hỗ trợ** hồ sơ *(thanh xanh trong biểu đồ SHAP)*:\n\n")
        for name, val in pos[:4]:
            strength = "rất mạnh" if abs(val) > 0.5 else "mạnh" if abs(val) > 0.25 else "vừa" if abs(val) > 0.1 else "nhẹ"
            lines.append(f"- **{_label(name)}** — tác động tích cực {strength} `(+{val:.3f})`\n")
        if neg:
            lines.append("\n---\n### ⚠️ Rủi ro tiềm ẩn (chưa đủ để từ chối)\n")
            lines.append("Những điểm này vẫn cần chú ý *(thanh đỏ)*:\n\n")
            for name, val in neg[:3]:
                lines.append(f"- **{_label(name)}** `({val:.3f})`\n")
            lines.append("\n💡 Cải thiện các yếu tố trên sẽ giúp tăng điểm tín dụng trong tương lai.\n")
    else:
        lines.append(f"## ❌ Khoản Vay BỊ TỪ CHỐI\n")
        lines.append(f"Xác suất nợ xấu: **{prob_default:.1%}** · Xác suất duyệt: **{prob_approve:.1%}**\n\n---\n")
        lines.append("### Tại sao bị từ chối?\n")
        lines.append("Các yếu tố sau đang **kéo hồ sơ xuống** *(thanh đỏ trong biểu đồ SHAP)*:\n\n")
        for name, val in neg[:4]:
            severity = "rất cao" if abs(val) > 0.5 else "cao" if abs(val) > 0.25 else "vừa"
            lines.append(f"- **{_label(name)}** — rủi ro {severity} `({val:.3f})`\n")
        if pos:
            lines.append("\n---\n### ✅ Điểm tích cực hiện có\n")
            lines.append("Những yếu tố này đang giúp hồ sơ, nhưng chưa đủ bù rủi ro:\n\n")
            for name, val in pos[:3]:
                lines.append(f"- **{_label(name)}** `(+{val:.3f})`\n")
        lines.append("\n---\n### 💡 Gợi ý cải thiện hồ sơ\n\n")
        advices = [a for a in (_advice(n) for n, _ in neg[:4]) if a]
        for a in (advices or ["Liên hệ chuyên viên tín dụng để được tư vấn cụ thể."]):
            lines.append(f"- {a}\n")

    return "".join(lines)


# ── Prediction function ───────────────────────────────────────────────────────
def predict_loan(age, income, home_ownership, emp_length, intent,
                 grade, amount, rate, percent_income, default_hist):
    raw = pd.DataFrame([[age, income, home_ownership, emp_length, intent,
                          grade, amount, rate, percent_income, default_hist]],
                        columns=['person_age', 'person_income', 'person_home_ownership',
                                 'person_emp_length', 'loan_intent', 'loan_grade',
                                 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                                 'cb_person_default_on_file'])
    processed     = preprocessor.transform(raw)
    feature_names = preprocessor.get_feature_names_out()
    input_df      = pd.DataFrame(processed, columns=feature_names)

    # class 1 = fully paid (approved), class 0 = default (rejected)
    prob_approve = float(model.predict_proba(input_df)[0][1])
    prob_default = 1.0 - prob_approve
    decision = "✅  LOAN APPROVED" if prob_approve > 0.5 else "❌  LOAN REJECTED"

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
#header { background: linear-gradient(135deg,#1e1b4b 0%,#0d1117 100%);
          border-radius:12px; padding:28px; margin-bottom:8px; }
.kpi-grid { display:grid; grid-template-columns:repeat(6,1fr);
            gap:10px; margin:12px 0 4px 0; }
.kpi-card { background:#161b22; border:1px solid #30363d;
            border-radius:10px; padding:18px 12px; text-align:center; }
.kpi-val  { font-size:1.7rem; font-weight:800; color:#e8b84b; margin:0; line-height:1.1; }
.kpi-lbl  { font-size:0.75rem; color:#9ca3af; margin-top:4px;
            text-transform:uppercase; letter-spacing:0.05em; }
.insight  { background:#161b22; border-left:4px solid #3b82f6;
            border-radius:6px; padding:14px 18px; margin:6px 0; color:#e5e7eb; font-size:0.92rem; }
#decision-box textarea { text-align:center !important; font-size:1.4rem !important;
                         font-weight:700 !important; padding:18px !important; }
#submit-btn { font-size:1.1rem !important; font-weight:600 !important; }
"""

# ── Gradio Blocks ─────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue",
                         font=gr.themes.GoogleFont("Inter")),
    title="AI Loan Approval System",
    css=CSS
) as iface:

    gr.HTML("""
        <div id="header">
            <h1 style="font-size:2rem;font-weight:800;margin:0;color:#fff;">🏦 AI Loan Approval System</h1>
            <p style="color:#9ca3af;font-size:0.95rem;margin:8px 0 0 0;">
                XGBoost · SHAP Explainability · EDA Dashboard · Giải thích tự động
            </p>
        </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — EDA Dashboard  (lazy-loaded via iface.load)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📊 Tổng Quan Dữ Liệu"):

            kpi_html = gr.HTML("<div style='text-align:center;color:#9ca3af;padding:20px'>Đang tải dữ liệu...</div>")

            with gr.Row():
                plot_donut = gr.Plot()
                plot_grade = gr.Plot()

            gr.HTML("""
            <div class="insight">
                💡 <strong>Insight:</strong> Hạng <strong>A</strong> có tỷ lệ duyệt >85%.
                Hạng <strong>E, F, G</strong> có rủi ro rất cao — đây là yếu tố phân biệt rõ nhất.
            </div>""")

            with gr.Row():
                plot_intent = gr.Plot()
                plot_own    = gr.Plot()

            gr.HTML("""
            <div class="insight">
                💡 <strong>Insight:</strong> Mục đích <strong>Kinh doanh (VENTURE)</strong> và
                <strong>Cải tạo nhà</strong> có tỷ lệ duyệt cao hơn vay cá nhân.
                Người <strong>sở hữu nhà (OWN)</strong> được duyệt nhiều hơn người thuê (RENT).
            </div>""")

            with gr.Row():
                plot_income = gr.Plot()
                plot_amount = gr.Plot()

            gr.HTML("""
            <div class="insight">
                💡 <strong>Insight:</strong> Hồ sơ được duyệt có <strong>thu nhập cao hơn</strong> và
                <strong>số tiền vay thấp hơn</strong> so với hồ sơ bị từ chối.
            </div>""")

            with gr.Row():
                plot_dti  = gr.Plot()
                plot_rate = gr.Plot()

            gr.HTML("""
            <div class="insight">
                💡 <strong>Insight:</strong> Nhóm nợ xấu có <strong>DTI ratio cao hơn rõ rệt</strong>
                (median ~0.35 so với ~0.18 nhóm duyệt) và <strong>lãi suất cao hơn</strong>.
                DTI > 0.4 là ngưỡng rủi ro cần cảnh báo.
            </div>""")

            with gr.Row():
                with gr.Column(scale=1):
                    plot_default = gr.Plot()
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="padding:20px;">
                        <h3 style="color:#e5e7eb;margin-bottom:12px;">📌 Key Insights Tóm Tắt</h3>
                        <div class="insight" style="border-color:#10b981;margin-bottom:8px;">
                            ✅ <strong>Hồ sơ tốt:</strong> Hạng A/B, thu nhập cao, DTI thấp,
                            sở hữu nhà, không có lịch sử nợ xấu
                        </div>
                        <div class="insight" style="border-color:#ef4444;margin-bottom:8px;">
                            ❌ <strong>Hồ sơ rủi ro:</strong> Hạng E–G, DTI > 0.4,
                            lãi suất > 15%, thuê nhà, có lịch sử nợ xấu
                        </div>
                        <div class="insight" style="border-color:#e8b84b;">
                            💡 <strong>Yếu tố quyết định nhất:</strong> Loan Grade, DTI Ratio,
                            và lịch sử tín dụng
                        </div>
                    </div>""")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — Loan Predictor
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🤖 Dự Đoán Khoản Vay"):

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Thông Tin Người Vay")
                    age = gr.Number(label="Tuổi (năm)", value=26, minimum=18, maximum=100)
                    income = gr.Number(label="Thu Nhập Hàng Năm ($)", value=44000)
                    emp_length = gr.Number(label="Thâm Niên Công Tác (năm)", value=6,
                                           minimum=0, maximum=50)
                    home_ownership = gr.Dropdown(
                        choices=["RENT", "OWN", "MORTGAGE", "OTHER"],
                        label="Hình Thức Nhà Ở", value="RENT")
                    default_history = gr.Radio(
                        choices=["N", "Y"], label="Lịch Sử Nợ Xấu?", value="N")

                with gr.Column(scale=1):
                    gr.Markdown("### 💳 Chi Tiết Khoản Vay")
                    amount = gr.Number(label="Số Tiền Vay ($)", value=11000)
                    rate = gr.Number(label="Lãi Suất (%)", value=10.37)
                    percent_income = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        label="Tỷ Lệ Nợ/Thu Nhập — DTI (0.26 = 26%)", value=0.26)
                    intent = gr.Dropdown(
                        choices=["PERSONAL", "EDUCATION", "MEDICAL",
                                 "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                        label="Mục Đích Vay", value="PERSONAL")
                    grade = gr.Dropdown(
                        choices=["A", "B", "C", "D", "E", "F", "G"],
                        label="Hạng Tín Dụng (A=Tốt nhất, G=Tệ nhất)", value="B")

            submit_btn = gr.Button("🔍  Phân Tích Hồ Sơ Vay", variant="primary",
                                   size="lg", elem_id="submit-btn")

            gr.HTML('<hr style="border-color:#30363d;margin:16px 0;"/>')
            gr.Markdown("### 📊 Kết Quả Phán Quyết")

            with gr.Row():
                decision_out = gr.Textbox(label="Quyết Định", interactive=False,
                                          elem_id="decision-box", scale=2)
                with gr.Column(scale=1):
                    prob_default_out = gr.Textbox(label="🔴 Xác Suất Nợ Xấu", interactive=False)
                    prob_approve_out = gr.Textbox(label="🟢 Xác Suất Được Duyệt", interactive=False)

            gr.HTML('<hr style="border-color:#30363d;margin:16px 0;"/>')

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Biểu Đồ SHAP — Phân Tích Tác Động")
                    gr.Markdown(
                        "> **🔵 Thanh xanh** → đẩy về phía **Phê Duyệt** *(giảm rủi ro)*  \n"
                        "> **🔴 Thanh đỏ** → đẩy về phía **Từ Chối** *(tăng rủi ro)*  \n"
                        "> Thanh càng dài = ảnh hưởng càng lớn lên quyết định"
                    )
                    shap_out = gr.Image(label="SHAP Waterfall Plot")

                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Giải Thích Tự Động")
                    explanation_out = gr.Markdown(
                        value="> *Nhấn **Phân Tích Hồ Sơ Vay** để xem giải thích chi tiết...*"
                    )

            gr.HTML('<hr style="border-color:#30363d;margin:16px 0;"/>')
            gr.Markdown("### 📋 Hồ Sơ Mẫu — Click để nạp")
            gr.Examples(
                examples=[
                    [32, 60000, "MORTGAGE", 12, "VENTURE",  "A", 5000,  7.50, 0.04, "N"],
                    [26, 44000, "RENT",      6, "PERSONAL", "B", 11000, 10.37, 0.26, "N"],
                    [22, 25000, "RENT",      1, "PERSONAL", "E", 15000, 18.50, 0.60, "Y"],
                ],
                inputs=[age, income, home_ownership, emp_length, intent,
                        grade, amount, rate, percent_income, default_history],
                label="✅ Low Risk (Grade A)  |  ⚠️ Borderline (Grade B)  |  ❌ High Risk (Grade E)"
            )

            submit_btn.click(
                fn=predict_loan,
                inputs=[age, income, home_ownership, emp_length, intent,
                        grade, amount, rate, percent_income, default_history],
                outputs=[decision_out, prob_default_out, prob_approve_out,
                         shap_out, explanation_out]
            )

    # ── Lazy-load EDA when the page opens in browser (not during build) ───────
    iface.load(
        fn=load_eda,
        outputs=[plot_donut, plot_grade, plot_intent, plot_own,
                 plot_income, plot_amount, plot_dti, plot_rate,
                 plot_default, kpi_html]
    )

if __name__ == "__main__":
    iface.launch()
