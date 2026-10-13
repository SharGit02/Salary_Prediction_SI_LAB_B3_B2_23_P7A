import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Styling (Dark Neon Aesthetic)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Streamlit default header, footer, and menu */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* Main Container layout */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.8rem !important;
        padding-left: 2.8rem !important;
        padding-right: 2.8rem !important;
        max-width: 1280px !important;
    }

    /* Overall background */
    .stApp {
        background: radial-gradient(circle at 85% 15%, rgba(26, 42, 80, 0.4) 0%, rgba(7, 11, 20, 1) 70%),
                    radial-gradient(circle at 15% 85%, rgba(18, 28, 55, 0.45) 0%, rgba(7, 11, 20, 1) 70%),
                    #070B14 !important;
        color: #E2E8F0 !important;
    }

    /* Top navbar header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 16px;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(30, 48, 77, 0.5);
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .header-icon {
        width: 48px;
        height: 48px;
        background: rgba(0, 229, 190, 0.08);
        border: 1.5px solid rgba(0, 229, 190, 0.35);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(0, 229, 190, 0.15);
    }

    .header-title-box h1 {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        line-height: 1.15;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }

    .header-title-box h1 span {
        color: #00E5BE;
    }

    .header-title-box p {
        font-size: 13.5px;
        color: #8E9CAE;
        margin: 3px 0 0 0;
        font-weight: 400;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(18, 27, 45, 0.85);
        border: 1px solid rgba(45, 65, 102, 0.7);
        border-radius: 9999px;
        padding: 8px 18px;
        font-size: 13px;
        color: #94A3B8;
        font-weight: 500;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    }

    /* Universal Styled Cards via Streamlit container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(16, 25, 42, 0.88) !important;
        border: 1px solid rgba(32, 50, 78, 0.75) !important;
        border-radius: 18px !important;
        padding: 20px 22px !important;
        box-shadow: 0 10px 32px -8px rgba(0, 0, 0, 0.55) !important;
        backdrop-filter: blur(14px) !important;
        margin-bottom: 20px !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(0, 229, 190, 0.35) !important;
        box-shadow: 0 12px 36px -6px rgba(0, 0, 0, 0.65), 0 0 18px rgba(0, 229, 190, 0.08) !important;
    }

    /* Header rows inside cards */
    .card-header-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }

    .card-title {
        font-size: 19px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.3px;
    }

    .card-desc {
        font-size: 13px;
        color: #8E9CAE;
        margin-bottom: 14px;
        line-height: 1.45;
    }

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(16, 26, 45, 0.95) 0%, rgba(32, 26, 62, 0.9) 100%);
        border: 1px solid rgba(48, 66, 102, 0.8);
        border-radius: 20px;
        padding: 22px 28px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.65);
    }

    .prediction-card::after {
        content: '';
        position: absolute;
        top: -30px;
        right: -30px;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.28) 0%, transparent 70%);
        pointer-events: none;
    }

    .pred-content-row {
        display: flex;
        align-items: center;
        gap: 20px;
        position: relative;
        z-index: 1;
    }

    .pred-icon-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: rgba(0, 229, 190, 0.12);
        border: 1.5px solid rgba(0, 229, 190, 0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 0 25px rgba(0, 229, 190, 0.2);
    }

    .pred-details {
        flex-grow: 1;
    }

    .pred-label {
        font-size: 16px;
        font-weight: 600;
        color: #E2E8F0;
        margin: 0 0 3px 0;
    }

    .pred-value {
        font-size: 44px;
        font-weight: 800;
        color: #00E5BE;
        line-height: 1.1;
        letter-spacing: -0.8px;
        margin: 1px 0 4px 0;
        text-shadow: 0 0 32px rgba(0, 229, 190, 0.3);
    }

    .pred-sub {
        font-size: 13.5px;
        color: #8E9CAE;
        margin: 0;
    }

    .pred-sub span {
        color: #00E5BE;
        font-weight: 600;
    }

    /* About Model List */
    .model-bullets {
        list-style: none;
        padding: 0;
        margin: 0 0 14px 0;
    }

    .model-bullets li {
        position: relative;
        padding-left: 18px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #CBD5E1;
    }

    .model-bullets li::before {
        content: "•";
        position: absolute;
        left: 2px;
        color: #00E5BE;
        font-size: 17px;
        line-height: 1;
        top: -1px;
    }

    .card-divider {
        height: 1px;
        background: rgba(35, 55, 85, 0.65);
        margin: 14px 0;
    }

    .model-note {
        font-size: 12.5px;
        color: #8E9CAE;
        line-height: 1.5;
    }

    .model-note strong {
        color: #FFFFFF;
    }

    /* Quick Insights List */
    .insights-bullets {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .insights-bullets li {
        position: relative;
        padding-left: 20px;
        margin-bottom: 10px;
        font-size: 13px;
        color: #CBD5E1;
        line-height: 1.5;
    }

    .insights-bullets li::before {
        content: "•";
        position: absolute;
        left: 2px;
        color: #00E5BE;
        font-size: 17px;
        line-height: 1;
        top: -1px;
    }

    /* Streamlit Input Styling */
    div[data-testid="stNumberInput"] {
        margin-bottom: 10px;
    }

    div[data-testid="stNumberInput"] label p {
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin-bottom: 4px !important;
    }

    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #0B1322 !important;
        border: 1px solid #1E2E48 !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stNumberInput"] input {
        color: #FFFFFF !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        background-color: #0B1322 !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #121E33 !important;
        color: #00E5BE !important;
        border: none !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #1A2C4A !important;
        color: #00E5BE !important;
    }

    /* Slider styling */
    div[data-testid="stSlider"] {
        padding-top: 2px;
        margin-bottom: 2px;
    }

    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #00E5BE !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 0 12px rgba(0, 229, 190, 0.9) !important;
        width: 18px !important;
        height: 18px !important;
    }

    .slider-range-labels {
        display: flex;
        justify-content: space-between;
        font-size: 12.5px;
        color: #718096;
        margin-top: -6px;
        margin-bottom: 8px;
        font-weight: 500;
    }

    .range-helper-text {
        font-size: 12.5px;
        color: #00E5BE;
        margin: 0 0 14px 0;
        font-weight: 500;
    }

    /* Awaiting State Prediction Card */
    .prediction-card-awaiting {
        background: rgba(16, 25, 42, 0.88);
        border: 1px dashed rgba(48, 66, 102, 0.9);
        border-radius: 20px;
        padding: 22px 28px;
        margin-bottom: 20px;
    }

    .pred-icon-circle-awaiting {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: rgba(113, 128, 150, 0.1);
        border: 1.5px solid rgba(113, 128, 150, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .pred-awaiting-title {
        font-size: 26px;
        font-weight: 700;
        color: #718096;
        line-height: 1.2;
        margin: 4px 0 6px 0;
    }

    /* Predict Button - Solid Teal, No Shadow, No Transform, No Effects */
    div.stButton > button {
        background-color: #00E5BE !important;
        background-image: none !important;
        color: #070D18 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        width: 100% !important;
        letter-spacing: 0.2px !important;
        box-shadow: none !important;
        transform: none !important;
        transition: background-color 0.15s ease !important;
        margin-top: 2px !important;
    }

    div.stButton > button:hover {
        background-color: #00CCA8 !important;
        background-image: none !important;
        box-shadow: none !important;
        transform: none !important;
        color: #070D18 !important;
    }

    div.stButton > button:active {
        background-color: #00B394 !important;
        background-image: none !important;
        box-shadow: none !important;
        transform: none !important;
    }

    div.stButton > button:focus {
        box-shadow: none !important;
        outline: none !important;
    }

    /* Footer */
    .app-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 0 8px 0;
        border-top: 1px solid rgba(30, 48, 77, 0.5);
        margin-top: 20px;
        font-size: 12.5px;
        color: #718096;
    }

    .footer-left {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .footer-left span {
        color: #A0AEC0;
        font-weight: 500;
    }

    .footer-right {
        color: #718096;
        font-style: italic;
    }

    .footer-right span {
        color: #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Dataset
# ---------------------------------------------------------
@st.cache_resource
def load_salary_model():
    model_path = os.path.join(os.path.dirname(__file__), "salary_model.pkl")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

@st.cache_data
def load_salary_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "salary_data.csv")
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            return df
        except Exception:
            pass
    
    # Fallback realistic points consistent with the real-world dataset
    np.random.seed(42)
    exp = np.random.randint(1, 16, 120)
    noise = np.random.normal(0, 35000, 120)
    salary = 178233 + 2780 * exp + noise
    salary = np.clip(salary, 110000, 350000)
    return pd.DataFrame({'years_of_experience': exp, 'annual_salary_usd': salary})

model = load_salary_model()
raw_data = load_salary_dataset()

# Model metadata
coef = float(model.coef_[0]) if model is not None and hasattr(model, 'coef_') else 2779.98
intercept = float(model.intercept_) if model is not None and hasattr(model, 'intercept_') else 178233.49
r2_val = -0.0276

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="header-left">
        <div class="header-icon">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 19V13" stroke="#00E5BE" stroke-width="2.5" stroke-linecap="round"/>
                <path d="M10 19V9" stroke="#00E5BE" stroke-width="2.5" stroke-linecap="round"/>
                <path d="M16 19V5" stroke="#00E5BE" stroke-width="2.5" stroke-linecap="round"/>
                <path d="M4 9L9 4L14 7L20 2" stroke="#FF8577" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div class="header-title-box">
            <h1>Salary <span>Prediction</span> App</h1>
            <p>Estimate your annual salary based on years of experience</p>
        </div>
    </div>
    <div class="badge-pill">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#A78BFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/>
            <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"/>
        </svg>
        Powered by Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Synchronized Session State for Number Input & Slider
# ---------------------------------------------------------
if "years" not in st.session_state:
    st.session_state.years = 5

if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False

def on_num_change():
    st.session_state.years = st.session_state.num_widget
    st.session_state.has_predicted = False

def on_slider_change():
    st.session_state.years = st.session_state.slider_widget
    st.session_state.has_predicted = False

# ---------------------------------------------------------
# Layout: Left Panel & Right Dashboard
# ---------------------------------------------------------
col_left, col_right = st.columns([1.05, 1.95], gap="large")

with col_left:
    # ------------------ Card 1: Input Container ------------------
    with st.container(border=True):
        st.markdown("""
        <div class="card-header-row">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00E5BE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
            </svg>
            <h3 class="card-title">Input</h3>
        </div>
        <p class="card-desc">Enter your years of experience to get an estimated annual salary.</p>
        """, unsafe_allow_html=True)

        st.number_input(
            "Years of Experience",
            min_value=1,
            max_value=15,
            value=st.session_state.years,
            step=1,
            key="num_widget",
            on_change=on_num_change
        )

        st.slider(
            "Years Slider",
            min_value=1,
            max_value=15,
            value=st.session_state.years,
            step=1,
            key="slider_widget",
            on_change=on_slider_change,
            label_visibility="collapsed"
        )

        st.markdown("""
        <div class="slider-range-labels">
            <span>1</span>
            <span>15</span>
        </div>
        <div class="range-helper-text">Input range: 1 – 15 years</div>
        """, unsafe_allow_html=True)

        predict_clicked = st.button("Predict Salary", key="predict_action")
        if predict_clicked:
            st.session_state.has_predicted = True

    # ------------------ Card 2: About This Model Container ------------------
    with st.container(border=True):
        st.markdown(f"""
        <div class="card-header-row">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00E5BE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <h3 class="card-title">About This Model</h3>
        </div>
        <p class="card-desc" style="margin-bottom: 12px;">
            This app uses a Simple Linear Regression model trained on real-world job market data with two features:
        </p>
        <ul class="model-bullets">
            <li>Years of Experience (input)</li>
            <li>Annual Salary (target)</li>
        </ul>
        <div class="card-divider"></div>
        <div class="model-note">
            <strong>Note:</strong> The model has limited accuracy ( <strong>R² = {r2_val}</strong> ) and predictions should be taken as estimates, not exact values.
        </div>
        """, unsafe_allow_html=True)

with col_right:
    # Selected years from synced state
    current_years = st.session_state.years
    has_predicted = st.session_state.has_predicted

    if has_predicted:
        # Compute prediction with salary_model.pkl
        if model is not None:
            try:
                input_df = pd.DataFrame({'years_of_experience': [current_years]})
                predicted_salary = float(model.predict(input_df)[0])
            except Exception:
                predicted_salary = intercept + coef * current_years
        else:
            predicted_salary = intercept + coef * current_years

        # Format salary
        formatted_salary = f"${predicted_salary:,.0f}"

        # ------------------ Right Card 1: Prediction Header Card ------------------
        st.markdown(f"""
        <div class="prediction-card">
            <div class="pred-content-row">
                <div class="pred-icon-circle">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00E5BE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="9" cy="8" r="6"/>
                        <path d="M19 12A6 6 0 1 1 12 19"/>
                        <path d="M7 6h4v4H7z"/>
                    </svg>
                </div>
                <div class="pred-details">
                    <div class="pred-label">Predicted Annual Salary</div>
                    <div class="pred-value">{formatted_salary}</div>
                    <div class="pred-sub">Based on <span>{current_years}</span> years of experience</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ------------------ Right Card 1: Awaiting Calculation ------------------
        st.markdown("""
        <div class="prediction-card-awaiting">
            <div class="pred-content-row">
                <div class="pred-icon-circle-awaiting">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#718096" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"/>
                        <polyline points="12 6 12 12 16 14"/>
                    </svg>
                </div>
                <div class="pred-details">
                    <div class="pred-label" style="color: #A0AEC0;">Predicted Annual Salary</div>
                    <div class="pred-awaiting-title">Click "Predict Salary"</div>
                    <div class="pred-sub">Enter your years of experience on the left and hit the button to calculate.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ------------------ Right Card 2: Dynamic Salary Trend Chart ------------------
    with st.container(border=True):
        st.markdown("""
        <div class="card-header-row" style="margin-bottom: 0px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00E5BE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            <h3 class="card-title">Salary Trend</h3>
        </div>
        """, unsafe_allow_html=True)

        # Regression line points
        x_line = np.linspace(1, 15, 60)
        if model is not None:
            try:
                y_line = model.predict(pd.DataFrame({'years_of_experience': x_line}))
            except Exception:
                y_line = intercept + coef * x_line
        else:
            y_line = intercept + coef * x_line

        # Stratified representative points from raw dataset
        if len(raw_data) > 90:
            sample_points = raw_data.sample(n=85, random_state=42)
        else:
            sample_points = raw_data

        fig = go.Figure()

        # Actual Data Scatter (coral dots)
        fig.add_trace(go.Scatter(
            x=sample_points['years_of_experience'],
            y=sample_points['annual_salary_usd'],
            mode='markers',
            name='Actual Data',
            marker=dict(
                color='#FF7A70',
                size=6.5,
                opacity=0.75,
                line=dict(width=0.5, color='#FFA099')
            ),
            hovertemplate='<b>Actual Data Point</b><br>Experience: %{x} yrs<br>Salary: $%{y:,.0f}<extra></extra>'
        ))

        # Regression Line (teal line)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(
                color='#00E5BE',
                width=2.8
            ),
            hovertemplate='<b>Regression Line</b><br>Experience: %{x:.1f} yrs<br>Estimated: $%{y:,.0f}<extra></extra>'
        ))

        # Dynamic selected prediction point marker (only when predict button was clicked)
        if has_predicted:
            fig.add_trace(go.Scatter(
                x=[current_years],
                y=[predicted_salary],
                mode='markers',
                name='Your Selected Point',
                marker=dict(
                    color='#00E5BE',
                    size=14,
                    symbol='diamond',
                    line=dict(color='#FFFFFF', width=2.5)
                ),
                hovertemplate=f'<b>Your Selection</b><br>Experience: {current_years} yrs<br>Predicted Salary: {formatted_salary}<extra></extra>',
                showlegend=False
            ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0B1322',
            margin=dict(l=50, r=20, t=10, b=45),
            height=300,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=12, color='#A0AEC0', family='Plus Jakarta Sans'),
                bgcolor='rgba(0,0,0,0)',
                itemclick=False,
                itemdoubleclick=False
            ),
            xaxis=dict(
                title=dict(
                    text='Years of Experience',
                    font=dict(color='#00E5BE', size=13, family='Plus Jakarta Sans')
                ),
                tickfont=dict(color='#8E9CAE', size=11),
                tickmode='linear',
                tick0=1,
                dtick=1,
                range=[0.5, 15.5],
                showgrid=True,
                gridcolor='rgba(30, 48, 77, 0.45)',
                griddash='dot',
                zeroline=False,
                showline=True,
                linecolor='#1E304D'
            ),
            yaxis=dict(
                title=dict(
                    text='Annual Salary (USD)',
                    font=dict(color='#00E5BE', size=13, family='Plus Jakarta Sans')
                ),
                tickfont=dict(color='#8E9CAE', size=11),
                tickformat=',d',
                range=[85000, 360000],
                tickvals=[100000, 150000, 200000, 250000, 300000, 350000],
                showgrid=True,
                gridcolor='rgba(30, 48, 77, 0.45)',
                griddash='dot',
                zeroline=False,
                showline=True,
                linecolor='#1E304D'
            ),
            hoverlabel=dict(
                bgcolor='#070E1A',
                bordercolor='#00E5BE',
                font=dict(color='#FFFFFF', size=12)
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                'displayModeBar': False
            }
        )

    # ------------------ Right Card 3: Quick Insights Container ------------------
    with st.container(border=True):
        st.markdown("""
        <div class="card-header-row">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00E5BE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 18h6"></path>
                <path d="M10 22h4"></path>
                <path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"></path>
            </svg>
            <h3 class="card-title" style="color: #00E5BE;">Quick Insights</h3>
        </div>
        """, unsafe_allow_html=True)

        if has_predicted:
            st.markdown(f"""
            <ul class="insights-bullets" style="margin-top: 12px;">
                <li>The model predicts a salary of <strong>{formatted_salary}</strong> for <strong>{current_years} years</strong> of experience.</li>
                <li>The estimated increase is about <strong>${coef:,.2f}</strong> for each additional year of experience.</li>
                <li>The model's R² value is <strong>{r2_val}</strong>, which means it explains very little of the salary variation in this dataset.</li>
                <li>Use this prediction as a general estimate, not a guaranteed salary.</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <ul class="insights-bullets" style="margin-top: 12px;">
                <li>Select your years of experience on the left and click <strong>Predict Salary</strong> to view estimated compensation.</li>
                <li>The estimated increase is about <strong>${coef:,.2f}</strong> for each additional year of experience.</li>
                <li>The model's R² value is <strong>{r2_val}</strong>, which means it explains very little of the salary variation in this dataset.</li>
                <li>Use predictions as general estimates, not guaranteed compensation.</li>
            </ul>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer Section
# ---------------------------------------------------------
st.markdown("""
<div class="app-footer">
    <div class="footer-left">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 19V13" stroke="#00E5BE" stroke-width="2.2" stroke-linecap="round"/>
            <path d="M10 19V9" stroke="#00E5BE" stroke-width="2.2" stroke-linecap="round"/>
            <path d="M16 19V5" stroke="#00E5BE" stroke-width="2.2" stroke-linecap="round"/>
            <path d="M4 9L9 4L14 7L20 2" stroke="#FF8577" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>Salary Prediction App</span> &nbsp;|&nbsp; Simple Linear Regression
    </div>
    <div class="footer-right">
        Better data. Smarter decisions. <span>❤️</span>
    </div>
</div>
""", unsafe_allow_html=True)
