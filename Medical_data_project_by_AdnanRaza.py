import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots



st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)





st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FF6B6B;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)




with st.sidebar:
    
    st.markdown("### Heart Disease Predictor")
    st.markdown("Health analysis dashboard")
    
    st.markdown("---")
    
    st.markdown("#### Settings")
    
    dataset_option = st.selectbox(
        "Dataset",
        ["Main Dataset", "Backup Dataset"]
    )
    
    models_to_use = st.multiselect(
        "ML Models",
        ["Logistic Regression", "KNN", "Both"],
        default=["Both"]
    )
    
    test_split = st.slider(
        "Test Size %",
        10, 40, 20
    )
    
    st.markdown("---")
    
    show_heatmap_chart = st.checkbox("Show Heatmap", True)
    show_histograms = st.checkbox("Show Histograms", True)
    show_scatter_plots = st.checkbox("Show Scatter Plots", True)
    
    st.markdown("---")
    
    if st.button("Start Analysis"):
        st.session_state.analysis_started = True
    else:
        if 'analysis_started' not in st.session_state:
            st.session_state.analysis_started = False



st.markdown('<div class="main-title">Heart Disease Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown("##### Machine Learning Application for Medical Analysis")






def get_heart_data():
    
    # first try the reliable github dataset (cleveland heart disease)
    main_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/heart.csv"
    
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    try:
        heart_df = pd.read_csv(main_url)
        
        if heart_df.shape[1] == 14:
            heart_df.columns = column_names
        else:
            # if different format, use backup
            raise Exception("Column mismatch")
        
        number_columns = heart_df.select_dtypes(include=[np.number]).columns
        for each_column in number_columns:
            if heart_df[each_column].isnull().any():
                heart_df[each_column].fillna(heart_df[each_column].median(), inplace=True)
        
        heart_df['target'] = heart_df['target'].apply(lambda val: 1 if val > 0 else 0)
        
        return heart_df
    
    except Exception as e:
        print("Main dataset failed, using backup")
        
        backup_url = "https://raw.githubusercontent.com/sid-krish/Heart-Disease-Prediction/main/data/heart.csv"
        
        backup_df = pd.read_csv(backup_url)
        
        if 'target' not in backup_df.columns:
            backup_df['target'] = backup_df['condition'] if 'condition' in backup_df.columns else backup_df.iloc[:,-1]
        
        backup_df['target'] = backup_df['target'].apply(lambda val: 1 if val > 0 else 0)
        
        number_cols = backup_df.select_dtypes(include=[np.number]).columns
        for col in number_cols:
            if backup_df[col].isnull().any():
                backup_df[col].fillna(backup_df[col].median(), inplace=True)
        
        return backup_df






with st.spinner("Loading dataset, please wait..."):
    data_frame = get_heart_data()




column1, column2, column3, column4 = st.columns(4)

with column1:
    total_records = len(data_frame)
    st.markdown(f"""
    <div style='background: white; padding: 1rem; border-radius: 10px; border-left: 5px solid #FF6B6B;'>
        <div style='font-size: 0.9rem; color: #666;'>Total Records</div>
        <div style='font-size: 1.8rem; font-weight: bold; color: #1E3A8A;'>{total_records}</div>
    </div>
    """, unsafe_allow_html=True)

with column2:
    disease_cases = data_frame['target'].sum()
    percentage_cases = (disease_cases/total_records)*100
    st.markdown(f"""
    <div style='background: white; padding: 1rem; border-radius: 10px; border-left: 5px solid #FF6B6B;'>
        <div style='font-size: 0.9rem; color: #666;'>Disease Cases</div>
        <div style='font-size: 1.8rem; font-weight: bold; color: #1E3A8A;'>{disease_cases}</div>
        <div style='font-size: 0.8rem; color: #666;'>{percentage_cases:.1f}% of total</div>
    </div>
    """, unsafe_allow_html=True)

with column3:
    features_count = len(data_frame.columns) - 1
    st.markdown(f"""
    <div style='background: white; padding: 1rem; border-radius: 10px; border-left: 5px solid #FF6B6B;'>
        <div style='font-size: 0.9rem; color: #666;'>Features</div>
        <div style='font-size: 1.8rem; font-weight: bold; color: #1E3A8A;'>{features_count}</div>
        <div style='font-size: 0.8rem; color: #666;'>medical parameters</div>
    </div>
    """, unsafe_allow_html=True)

with column4:
    mean_age = data_frame['age'].mean()
    st.markdown(f"""
    <div style='background: white; padding: 1rem; border-radius: 10px; border-left: 5px solid #FF6B6B;'>
        <div style='font-size: 0.9rem; color: #666;'>Average Age</div>
        <div style='font-size: 1.8rem; font-weight: bold; color: #1E3A8A;'>{mean_age:.1f}</div>
        <div style='font-size: 0.8rem; color: #666;'>years</div>
    </div>
    """, unsafe_allow_html=True)




st.markdown('<div class="section-title">Data Overview</div>', unsafe_allow_html=True)

tab_one, tab_two, tab_three, tab_four = st.tabs(["First Look", "Stats", "Disease Split", "Missing Data"])

with tab_one:
    st.write(data_frame.head(10))
    st.text(f"First 10 records from {len(data_frame)} total")

with tab_two:
    st.write(data_frame.describe())

with tab_three:
    pie_chart = px.pie(data_frame, names='target', title='Heart Disease Distribution')
    pie_chart.update_traces(textposition='inside')
    pie_chart.update_layout(height=400)
    st.plotly_chart(pie_chart, use_container_width=True)

with tab_four:
    missing_info = pd.DataFrame({
        'Column': data_frame.columns,
        'Missing': data_frame.isnull().sum().values,
        'Percent': (data_frame.isnull().sum() / len(data_frame) * 100).round(2)
    })
    missing_chart = px.bar(missing_info, x='Column', y='Percent', title='Missing Values')
    missing_chart.update_layout(height=400)
    st.plotly_chart(missing_chart, use_container_width=True)




if st.session_state.analysis_started or st.button("Run Data Analysis"):
    st.markdown('<div class="section-title">Data Analysis</div>', unsafe_allow_html=True)
    
    if show_heatmap_chart:
        st.write("Correlation between features")
        correlation_matrix = data_frame.corr()
        
        heatmap_figure = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            hoverongaps=False
        ))
        
        heatmap_figure.update_layout(height=500, title="Correlation Heatmap")
        st.plotly_chart(heatmap_figure, use_container_width=True)
        
        top_correlations = correlation_matrix['target'].sort_values(ascending=False)[1:6]
        columns_for_corr = st.columns(5)
        for i, (feature_name, corr_value) in enumerate(top_correlations.items()):
            with columns_for_corr[i]:
                st.metric(label=feature_name, value=f"{corr_value:.3f}")
    
    if show_histograms:
        st.write("Distribution of features")
        
        selected_features = st.multiselect(
            "Choose features to plot:",
            options=data_frame.columns.tolist(),
            default=['age', 'chol', 'thalach', 'trestbps'],
            max_selections=6
        )
        
        if selected_features:
            rows_count = (len(selected_features) + 1) // 2
            cols_count = 2
            
            distribution_figure = make_subplots(
                rows=rows_count,
                cols=cols_count,
                subplot_titles=selected_features
            )
            
            for index, feature in enumerate(selected_features):
                row_num = (index // cols_count) + 1
                col_num = (index % cols_count) + 1
                
                distribution_figure.add_trace(
                    go.Histogram(x=data_frame[feature], name=feature),
                    row=row_num, col=col_num
                )
            
            distribution_figure.update_layout(height=300 * rows_count, showlegend=False)
            st.plotly_chart(distribution_figure, use_container_width=True)
    
    if show_scatter_plots:
        st.write("Scatter plot analysis")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            x_feature = st.selectbox("X feature", data_frame.columns.tolist(), index=0)
        with col_b:
            y_feature = st.selectbox("Y feature", data_frame.columns.tolist(), index=7)
        with col_c:
            color_feature = st.selectbox("Color by", ['target', 'sex', 'cp'], index=0)
        
        scatter_plot = px.scatter(
            data_frame,
            x=x_feature,
            y=y_feature,
            color=color_feature,
            hover_data=data_frame.columns.tolist(),
            title=f"{x_feature} vs {y_feature}"
        )
        
        scatter_plot.update_layout(height=500)
        st.plotly_chart(scatter_plot, use_container_width=True)




st.markdown('<div class="section-title">Machine Learning Models</div>', unsafe_allow_html=True)




X_data = data_frame.drop('target', axis=1)
y_data = data_frame['target']

X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
    X_data, y_data, test_size=test_split/100, random_state=42, stratify=y_data
)

data_scaler = StandardScaler()
X_train_scaled = data_scaler.fit_transform(X_train_data)
X_test_scaled = data_scaler.transform(X_test_data)




model_dict = {}
prediction_dict = {}

if "Logistic Regression" in models_to_use or "Both" in models_to_use:
    with st.spinner("Working on Logistic Regression..."):
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train_data)
        model_dict['Logistic Regression'] = lr_model
        prediction_dict['Logistic Regression'] = lr_model.predict(X_test_scaled)

if "KNN" in models_to_use or "Both" in models_to_use:
    with st.spinner("Working on KNN..."):
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train_scaled, y_train_data)
        model_dict['K-Nearest Neighbors'] = knn_model
        prediction_dict['K-Nearest Neighbors'] = knn_model.predict(X_test_scaled)




if model_dict:
    st.write("Model Performance")
    
    performance_data = []
    for model_name, predictions in prediction_dict.items():
        acc = accuracy_score(y_test_data, predictions)
        f1_val = f1_score(y_test_data, predictions)
        prec = precision_score(y_test_data, predictions)
        
        performance_data.append({
            'Model': model_name,
            'Accuracy': acc,
            'F1': f1_val,
            'Precision': prec
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    display_cols = st.columns(len(model_dict))
    for idx, (m_name, row) in enumerate(zip(model_dict.keys(), performance_df.itertuples())):
        with display_cols[idx]:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white;'>
                <h4>{m_name}</h4>
                <p>Accuracy: <b>{row.Accuracy:.2%}</b></p>
                <p>F1 Score: <b>{row.F1:.3f}</b></p>
                <p>Precision: <b>{row.Precision:.3f}</b></p>
            </div>
            """, unsafe_allow_html=True)
    
    bar_figure = go.Figure()
    
    bar_figure.add_trace(go.Bar(
        name='Accuracy',
        x=performance_df['Model'],
        y=performance_df['Accuracy'],
        text=performance_df['Accuracy'].round(3)
    ))
    
    bar_figure.add_trace(go.Bar(
        name='F1 Score',
        x=performance_df['Model'],
        y=performance_df['F1'],
        text=performance_df['F1'].round(3)
    ))
    
    bar_figure.add_trace(go.Bar(
        name='Precision',
        x=performance_df['Model'],
        y=performance_df['Precision'],
        text=performance_df['Precision'].round(3)
    ))
    
    bar_figure.update_layout(
        barmode='group',
        title="Performance Comparison",
        height=400
    )
    st.plotly_chart(bar_figure, use_container_width=True)
    
    st.write("Confusion Matrices")
    
    num_models = len(model_dict)
    confusion_figure = make_subplots(
        rows=1,
        cols=num_models,
        subplot_titles=list(model_dict.keys())
    )
    
    for i, (m_name, preds) in enumerate(prediction_dict.items(), 1):
        cm_matrix = confusion_matrix(y_test_data, preds)
        
        confusion_figure.add_trace(
            go.Heatmap(
                z=cm_matrix,
                x=['No Disease', 'Disease'],
                y=['Actual No', 'Actual Yes'],
                text=cm_matrix,
                texttemplate="%{text}",
                colorscale='Blues',
                showscale=False
            ),
            row=1, col=i
        )
    
    confusion_figure.update_layout(height=350)
    st.plotly_chart(confusion_figure, use_container_width=True)
    
    if "Logistic Regression" in model_dict:
        st.write("Feature Importance")
        
        importance_values = pd.DataFrame({
            'Feature': X_data.columns,
            'Importance': abs(model_dict['Logistic Regression'].coef_[0])
        }).sort_values('Importance', ascending=True)
        
        importance_chart = px.bar(
            importance_values,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance"
        )
        
        importance_chart.update_layout(height=400)
        st.plotly_chart(importance_chart, use_container_width=True)




st.markdown('<div class="section-title">Patient Risk Assessment</div>', unsafe_allow_html=True)

st.markdown("Enter patient information below for prediction")




c1, c2, c3, c4 = st.columns(4)

with c1:
    patient_age = st.slider("Patient Age", 20, 100, 55)
    patient_sex = st.selectbox("Gender", ["Male", "Female"])
    chest_pain = st.selectbox(
        "Chest Pain",
        ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
    )

with c2:
    resting_bp = st.slider("Blood Pressure", 90, 200, 130)
    cholesterol = st.slider("Cholesterol", 100, 600, 250)
    blood_sugar = st.selectbox("High Blood Sugar", ["No", "Yes"])

with c3:
    resting_ecg = st.selectbox(
        "ECG Results",
        ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
    )
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])

with c4:
    st_depression = st.slider("ST Depression", 0.0, 10.0, 1.0, 0.1)
    st_slope = st.selectbox(
        "ST Slope",
        ["Upsloping", "Flat", "Downsloping"]
    )
    major_vessels = st.slider("Major Vessels", 0, 4, 1)




patient_input = {
    'age': patient_age,
    'sex': 1 if patient_sex == "Male" else 0,
    'cp': ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"].index(chest_pain),
    'trestbps': resting_bp,
    'chol': cholesterol,
    'fbs': 1 if blood_sugar == "Yes" else 0,
    'restecg': ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(resting_ecg),
    'thalach': max_hr,
    'exang': 1 if exercise_angina == "Yes" else 0,
    'oldpeak': st_depression,
    'slope': ["Upsloping", "Flat", "Downsloping"].index(st_slope) + 1,
    'ca': major_vessels,
    'thal': 3
}




input_df = pd.DataFrame([patient_input])
scaled_input = data_scaler.transform(input_df)




if st.button("Check Risk Level"):
    if model_dict:
        risk_results = {}
        
        for m_name, m_model in model_dict.items():
            risk_prob = m_model.predict_proba(scaled_input)[0][1]
            risk_results[m_name] = risk_prob * 100
        
        st.markdown("---")
        
        average_risk = sum(risk_results.values()) / len(risk_results)
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            if average_risk > 50:
                bg_color = '#FFE5E5'
                border_color = '#FF6B6B'
                text_color = '#FF4B4B'
                risk_level = "HIGH RISK"
            else:
                bg_color = '#E5FFE5'
                border_color = '#4CAF50'
                text_color = '#2E7D32'
                risk_level = "LOW RISK"
            
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: {bg_color};
                     border-radius: 15px; border: 3px solid {border_color}; margin-bottom: 1rem;'>
                <h2 style='color: {text_color};'>{risk_level}</h2>
                <p style='font-size: 1.2rem;'>Predicted Probability</p>
                <h1 style='font-size: 3.5rem; color: {text_color};'>{average_risk:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown("**Model Predictions:**")
            for m_name, risk_val in risk_results.items():
                st.write(f"{m_name}: {risk_val:.1f}%")
        
        if average_risk > 50:
            st.warning("Consult a healthcare provider for further evaluation")
        else:
            st.success("Low risk detected. Maintain healthy lifestyle")