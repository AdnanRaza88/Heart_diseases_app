

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS FOR BEAUTIFUL UI ============
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
        border-bottom: 3px solid #FF6B6B;
        padding-bottom: 0.5rem;
    }
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #FF6B6B;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(255,107,107,0.3);
    }
    .feature-box {
        background: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #E9ECEF;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #FF4B4B;'>❤️ Heart Disease Predictor</h2>
        <p>AI-powered health analysis dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🎛️ Control Panel")
    dataset_choice = st.selectbox(
        "Choose Dataset",
        ["UCI Heart Disease", "Kaggle Heart Disease"],
        help="Select the dataset for analysis"
    )

    model_choice = st.multiselect(
        "Select ML Models",
        ["Logistic Regression", "K-Nearest Neighbors", "Both"],
        default=["Both"],
        help="Choose which models to train and compare"
    )

    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        help="Percentage of data for testing"
    )

    st.markdown("---")
    st.markdown("### 📊 Visualization Settings")
    show_heatmap = st.checkbox("Show Correlation Heatmap", value=True)
    show_distributions = st.checkbox("Show Feature Distributions", value=True)
    show_scatter = st.checkbox("Show Scatter Plots", value=True)

    st.markdown("---")
    if st.button("🚀 Run Complete Analysis", use_container_width=True):
        st.session_state.run_analysis = True
    else:
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False

# ============ MAIN PAGE ============
st.markdown("<h1 class='main-header'>❤️ Heart Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### _Advanced Machine Learning with Beautiful Visualizations_")

# ============ LOAD DATASET ============
@st.cache_data
def load_data():
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, names=column_names, na_values='?')

        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Convert target to binary
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

        return df
    except:
        # Backup dataset
        backup_url = "https://raw.githubusercontent.com/sid-krish/Heart-Disease-Prediction/main/data/heart.csv"
        df = pd.read_csv(backup_url)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        return df

# Load data
with st.spinner("🔄 Loading dataset..."):
    df = load_data()

# ============ DASHBOARD METRICS ============
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #FF6B6B; margin: 0;'>📊 Total Samples</h3>
        <p style='font-size: 2rem; font-weight: 700; color: #1E3A8A;'>{}</p>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

with col2:
    positive_cases = df['target'].sum()
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #FF6B6B; margin: 0;'>❤️ Heart Disease Cases</h3>
        <p style='font-size: 2rem; font-weight: 700; color: #1E3A8A;'>{}</p>
        <p style='color: #666;'>{:.1f}% of total</p>
    </div>
    """.format(positive_cases, (positive_cases/len(df))*100), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #FF6B6B; margin: 0;'>🔢 Features</h3>
        <p style='font-size: 2rem; font-weight: 700; color: #1E3A8A;'>{}</p>
        <p style='color: #666;'>Clinical parameters</p>
    </div>
    """.format(len(df.columns) - 1), unsafe_allow_html=True)

with col4:
    avg_age = df['age'].mean()
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: #FF6B6B; margin: 0;'>📈 Average Age</h3>
        <p style='font-size: 2rem; font-weight: 700; color: #1E3A8A;'>{:.1f}</p>
        <p style='color: #666;'>Years</p>
    </div>
    """.format(avg_age), unsafe_allow_html=True)

# ============ DATASET PREVIEW ============
st.markdown("<h2 class='sub-header'>📋 Dataset Overview</h2>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "📈 Statistics", "🎯 Target Distribution", "🔍 Missing Values"])

with tab1:
    st.dataframe(df.head(10), width='stretch')
    st.caption(f"Showing 10 of {len(df)} records")

with tab2:
    st.dataframe(df.describe(), width='stretch')

with tab3:
    fig = px.pie(df, names='target',
                 title='Heart Disease Distribution',
                 color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')

with tab4:
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum().values,
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    fig = px.bar(missing_df, x='Column', y='Percentage',
                 title='Missing Values Percentage by Column',
                 color='Percentage',
                 color_continuous_scale='Reds')
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')

# ============ EXPLORATORY DATA ANALYSIS ============
if st.session_state.run_analysis or st.button("📊 Run EDA Analysis"):
    st.markdown("<h2 class='sub-header'>🔬 Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    # 1. CORRELATION HEATMAP
    if show_heatmap:
        st.markdown("#### 🔥 Correlation Heatmap")
        corr_matrix = df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            height=600,
            title="Feature Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features"
        )
        st.plotly_chart(fig, width='stretch')

        # Highlight top correlations
        st.markdown("##### 🎯 Top Correlations with Target")
        target_corr = corr_matrix['target'].sort_values(ascending=False)[1:6]
        cols = st.columns(5)
        for idx, (feature, corr) in enumerate(target_corr.items()):
            with cols[idx]:
                color = "🟢" if corr > 0 else "🔴"
                st.metric(label=f"{color} {feature}", value=f"{corr:.3f}")

    # 2. FEATURE DISTRIBUTIONS
    if show_distributions:
        st.markdown("#### 📊 Feature Distributions")

        # Select features to visualize
        features_to_plot = st.multiselect(
            "Select features for distribution plots:",
            options=df.columns.tolist(),
            default=['age', 'chol', 'thalach', 'trestbps'],
            max_selections=6
        )

        if features_to_plot:
            n_cols = 2
            n_rows = (len(features_to_plot) + 1) // n_cols

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=features_to_plot
            )

            for idx, feature in enumerate(features_to_plot):
                row = (idx // n_cols) + 1
                col = (idx % n_cols) + 1

                # Add histogram for each feature
                fig.add_trace(
                    go.Histogram(
                        x=df[feature],
                        name=feature,
                        marker_color='#FF6B6B',
                        opacity=0.7
                    ),
                    row=row, col=col
                )

            fig.update_layout(
                height=300 * n_rows,
                showlegend=False,
                title_text="Feature Distributions"
            )
            st.plotly_chart(fig, width='stretch')

    # 3. SCATTER PLOTS
    if show_scatter:
        st.markdown("#### 📍 Interactive Scatter Plots")

        col1, col2, col3 = st.columns(3)

        with col1:
            x_axis = st.selectbox("X-axis feature", df.columns.tolist(), index=0)
        with col2:
            y_axis = st.selectbox("Y-axis feature", df.columns.tolist(), index=7)
        with col3:
            color_by = st.selectbox("Color by", ['target', 'sex', 'cp'], index=0)

        fig = px.scatter(df,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        size='age',
                        hover_data=df.columns.tolist(),
                        title=f"{x_axis} vs {y_axis} (Colored by {color_by})",
                        color_continuous_scale=px.colors.sequential.Viridis)

        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')

# ============ MACHINE LEARNING MODELS ============
st.markdown("<h2 class='sub-header'>🤖 Machine Learning Models</h2>", unsafe_allow_html=True)

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {}
predictions = {}

if "Logistic Regression" in model_choice or "Both" in model_choice:
    with st.spinner("Training Logistic Regression..."):
        log_model = LogisticRegression(max_iter=1000, random_state=42)
        log_model.fit(X_train_scaled, y_train)
        models['Logistic Regression'] = log_model
        predictions['Logistic Regression'] = log_model.predict(X_test_scaled)

if "K-Nearest Neighbors" in model_choice or "Both" in model_choice:
    with st.spinner("Training KNN..."):
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train_scaled, y_train)
        models['K-Nearest Neighbors'] = knn_model
        predictions['K-Nearest Neighbors'] = knn_model.predict(X_test_scaled)

# ============ MODEL EVALUATION ============
if models:
    st.markdown("#### 📈 Model Performance Comparison")

    # Performance metrics
    metrics_data = []
    for model_name, y_pred in predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        metrics_data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1_Score': f1, # Changed key to F1_Score
            'Precision': precision
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Display metrics in columns
    cols = st.columns(len(models))
    for idx, (model_name, row) in enumerate(zip(models.keys(), metrics_df.itertuples())):
        with cols[idx]:
            st.markdown(f"""
            <div class='card'>
                <h4>{model_name}</h4>
                <p>🎯 Accuracy: <b>{row.Accuracy:.2%}</b></p>
                <p>⚖️ F1-Score: <b>{row.F1_Score:.3f}</b></p>
                <p>📏 Precision: <b>{row.Precision:.3f}</b></p>
            </div>
            """, unsafe_allow_html=True)

    # Visualization
    fig = go.Figure()

    for metric in ['Accuracy', 'F1-Score', 'Precision']:
        # Note: 'F1-Score' is still used here as it's directly accessing the DataFrame column
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric if metric != 'F1-Score' else 'F1_Score'], # Accessing correct column name for F1-Score in df
            text=metrics_df[metric if metric != 'F1-Score' else 'F1_Score'].round(3),
            textposition='auto',
        ))

    fig.update_layout(
        barmode='group',
        title="Model Performance Metrics",
        xaxis_title="Models",
        yaxis_title="Score",
        height=400,
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig, width='stretch')

    # ============ CONFUSION MATRICES ============
    st.markdown("#### 🎯 Confusion Matrices")

    n_models = len(models)
    fig = make_subplots(
        rows=1,
        cols=n_models,
        subplot_titles=list(models.keys())
    )

    for idx, (model_name, y_pred) in enumerate(predictions.items(), 1):
        cm = confusion_matrix(y_test, y_pred)

        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted No', 'Predicted Yes'],
                y=['Actual No', 'Actual Yes'],
                text=cm,
                texttemplate="%{text}",
                colorscale='Blues',
                showscale=False,
                hoverinfo='z'
            ),
            row=1, col=idx
        )

        fig.update_xaxes(title_text="Predicted", row=1, col=idx)
        fig.update_yaxes(title_text="Actual", row=1, col=idx)

    fig.update_layout(height=400, title_text="Confusion Matrices")
    st.plotly_chart(fig, width='stretch')

    # ============ FEATURE IMPORTANCE ============
    if "Logistic Regression" in models:
        st.markdown("#### 🏆 Feature Importance (Logistic Regression)")

        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': abs(models['Logistic Regression'].coef_[0])
        }).sort_values('Importance', ascending=True)

        fig = px.bar(importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance Scores",
                    color='Importance',
                    color_continuous_scale='Viridis')

        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

# ============ LIVE PREDICTION ============
st.markdown("<h2 class='sub-header'>🎯 Live Heart Disease Prediction</h2>", unsafe_allow_html=True)

st.markdown("""
<div class='feature-box'>
    <h4>📝 Enter Patient Details</h4>
    <p>Fill in the clinical parameters below to get a real-time prediction</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.slider("👤 Age", 20, 100, 55)
    sex = st.selectbox("⚤ Sex", ["Male", "Female"])
    cp = st.selectbox("💔 Chest Pain Type",
                      ["Typical Angina", "Atypical Angina",
                       "Non-anginal Pain", "Asymptomatic"])

with col2:
    trestbps = st.slider("🩸 Resting BP (mm Hg)", 90, 200, 130)
    chol = st.slider("🧪 Cholesterol (mg/dl)", 100, 600, 250)
    fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col3:
    restecg = st.selectbox("📈 Resting ECG",
                          ["Normal", "ST-T Wave Abnormality",
                           "Left Ventricular Hypertrophy"])
    thalach = st.slider("❤️ Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("😓 Exercise Induced Angina", ["No", "Yes"])

with col4:
    oldpeak = st.slider("📉 ST Depression", 0.0, 10.0, 1.0, 0.1)
    slope = st.selectbox("📊 ST Slope",
                        ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("🧬 Number of Major Vessels", 0, 4, 1)

# Convert inputs to model format
input_data = {
    'age': age,
    'sex': 1 if sex == "Male" else 0,
    'cp': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp),
    'trestbps': trestbps,
    'chol': chol,
    'fbs': 1 if fbs == "Yes" else 0,
    'restecg': ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg),
    'thalach': thalach,
    'exang': 1 if exang == "Yes" else 0,
    'oldpeak': oldpeak,
    'slope': ["Upsloping", "Flat", "Downsloping"].index(slope) + 1,
    'ca': ca,
    'thal': 3  # Default value
}

# Prepare input array
input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# Make prediction
if st.button("🔮 Predict Heart Disease Risk", use_container_width=True):
    if models:
        # Get predictions from all trained models
        results = {}
        for model_name, model in models.items():
            prob = model.predict_proba(input_scaled)[0][1]
            results[model_name] = prob * 100

        # Display results
        st.markdown("---")

        # Calculate average probability
        avg_prob = sum(results.values()) / len(results)

        # Create columns for results
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: {'#FFE5E5' if avg_prob > 50 else '#E5FFE5'};
                     border-radius: 15px; border: 3px solid {'#FF6B6B' if avg_prob > 50 else '#4CAF50'};'>
                <h2 style='color: {'#FF4B4B' if avg_prob > 50 else '#2E7D32'};'>
                    {'⚠️ HIGH RISK' if avg_prob > 50 else '✅ LOW RISK'}
                </h2>
                <p style='font-size: 1.2rem;'>Predicted Probability of Heart Disease:</p>
                <h1 style='font-size: 4rem; color: {'#FF4B4B' if avg_prob > 50 else '#2E7D32'};'>{avg_prob:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='feature-box'>
                <h4>Model Probabilities</h4>
            </div>
            """, unsafe_allow_html=True)
            for model_name, prob in results.items():
                st.info(f"**{model_name}**: {prob:.2f}%")
