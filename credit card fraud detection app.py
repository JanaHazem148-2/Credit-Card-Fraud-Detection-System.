import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from io import StringIO
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------
# 🌌 FUTURISTIC THEME SETUP
# ------------------------------
st.set_page_config(
    page_title="NeoFraud | AI-Powered Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-modern look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        width:350;
    }
    .st-emotion-cache-1aehpvj {
    color: rgba(255, 255, 255, 0.6);
    font-size: 14px;
    line-height: 1.25;
}
            
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 12px 24px !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #6e45e2, #88d3ce) !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6e45e2, #88d3ce) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(110, 69, 226, 0.4) !important;
    }
    
    /* Inputs */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stMultiselect>div>div>div {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 10px 15px !important;
    }
    
    /* Tables */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Expanders */
    .stExpander {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        text-shadow: 0 2px 10px rgba(110, 69, 226, 0.3);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(#6e45e2, #88d3ce);
        border-radius: 10px;
    }
    
    /* Custom container styles */
    .custom-container {
        background: rgba(110, 69, 226, 0.1);
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .feature-card {
        background: rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
        height: 375px;
       
        width: 100%;  
        max-width: 300px;    
        word-wrap: break-word; 
        box-sizing: border-box
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .feature-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 20px;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        background: rgba(110, 69, 226, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .custom-label {
          color: #1f77b4;  /* Choose any color: hex, name, rgb */
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
   
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 🛠 UTILITY FUNCTIONS
# ------------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None
def styled_metric(label, value, color):
    st.markdown(f"""
        <div style="text-align:center; padding:auto;">
            <div style="font-size:18px; color:white;">{label}</div>
            <div style="font-size:24px; color:{color}; font-weight:bold;">{value}</div>
        </div>
    """, unsafe_allow_html=True)
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Genuine', 'Fraud'],
        y=['Genuine', 'Fraud'],
        colorscale='Blues',
        hoverongaps=False,
        text=cm,
        texttemplate="%{text}",
        showscale=False
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white')),
        xaxis_title='Predicted',
        yaxis_title='Actual',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=50, r=50, b=50, t=50),
        height=400
    )
    return fig

def plot_metrics_comparison(comparison_df):
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)']
    colors = px.colors.qualitative.Plotly
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=comparison_df['Model'],
            y=comparison_df[metric],
            name=metric,
            marker_color=colors[i],
            text=comparison_df[metric].round(2),
            textposition='outside',
            hovertemplate=f"{metric}: %{{y:.2f}}<extra></extra>"
        ))
    
    fig.update_layout(
        barmode='group',
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        yaxis_range=[0, 1.1],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=14,
            font_family="Arial"
        ),
        margin=dict(l=50, r=50, b=50, t=50),
        height=500
    )
    
    return fig

# ------------------------------
# 🎨 SIDEBAR DESIGN
# ------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="background: linear-gradient(90deg, #6e45e2, #88d3ce);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   margin: 0;">NEO FRAUD</h1>
        <p style="color: #aaa; margin-top: 5px;">AI-Powered Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-label">📤 Upload Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file)
        
        if 'Class' not in data.columns:
            st.error("Dataset must contain 'Class' column")
            st.stop()
            
        st.markdown("### 🔍 Dataset Insights")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            styled_metric("Total Records", len(data), "white")
        with col2:
            styled_metric("Fraud Cases", data['Class'].sum(), "white")
        with col3:
            fraud_rate = f"{data['Class'].mean()*100:.2f}%"
            styled_metric("Fraud Rate", fraud_rate,"white")
        
        st.markdown("---")
        
        st.markdown("### ⚙ Model Selection")
        st.markdown('<label style="color:white; font-weight:500;">Select models to run:</label>', unsafe_allow_html=True)
        models_to_run = st.multiselect(
            label="",  # Empty label since we styled it above
            options=["Naive Bayes", "KNN", "AdaBoost", "Decision Tree"],
            default=[]
        )

        
        st.markdown("""
    <style>
        /* Make dataframe background white */
        .stDataFrame div[data-testid="stHorizontalBlock"] {
            background-color: white !important;
            border-radius: 5px;
            padding: 10px;
        }

        /* Optional: Make table text darker for contrast */
        .stDataFrame td {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Checkbox to show data
        if st.checkbox("🔎 Show Data Explorer"):
            st.dataframe(data, use_container_width=True)
# ------------------------------
# 🖥 MAIN CONTENT
# ------------------------------
if not uploaded_file:
    # Hero section when no file uploaded
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="margin-top: 100px;">
            <h1 style="font-size: 48px; margin-bottom: 20px;">AI-POWERED <span style="color: #6e45e2;">FRAUD</span> DETECTION</h1>
            <p style="font-size: 18px; color: #aaa; margin-bottom: 30px;">
                Advanced machine learning models to detect fraudulent transactions with 
                state-of-the-art accuracy and performance.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("""
    <h2 style="text-align: center; margin-bottom: 40px;">✨ WHY CHOOSE NEO FRAUD?</h2>
    """, unsafe_allow_html=True)
    
    features = [
        {"icon": "⚡", "title": "Lightning Fast", "desc": "Real-time fraud detection with sub-second latency"},
        {"icon": "🔍", "title": "Precision AI", "desc": "Advanced algorithms with 99.9% accuracy"},
        {"icon": "🔄", "title": "Continuous Learning", "desc": "Models that improve over time"},
        {"icon": "📊", "title": "Visual Analytics", "desc": "Beautiful interactive dashboards"}
    ]
    
    cols = st.columns(4)
    for i, feature in enumerate(features):
        with cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <h1 style='font-size: 36px; margin-bottom: 10px;'>{feature['icon']}</h1>
                <h3 style="color: #6e45e2; margin-top: 0;">{feature['title']}</h3>
                <p style='color: #aaa;'>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.stop()

# ------------------------------
#   MODEL IMPLEMENTATIONS
# ------------------------------
tab1, tab2 = st.tabs(["🧪 Model Laboratory", "📊 Performance Dashboard"])

with tab1:
    if not models_to_run:
        st.info("ℹ Please select at least one model from the sidebar to run")
    else:
        # Initialize results storage
        if 'results' not in st.session_state:
            st.session_state.results = {}
        if 'model_order' not in st.session_state:
            st.session_state.model_order = models_to_run.copy()
        for model_name in list(st.session_state.results.keys()):
            if model_name not in models_to_run:
                del st.session_state.results[model_name]
        st.session_state.model_order = [
            m for m in st.session_state.model_order 
            if m in models_to_run
        ] + [
            m for m in models_to_run 
            if m not in st.session_state.model_order
        ]

        # Naive Bayes
        for model_name in st.session_state.model_order:
            if model_name == "Naive Bayes":
        
                with st.expander("🧪 Gaussian Naive Bayes", expanded=True):
                    st.markdown("""
                    <div class="custom-container">
                        <h3 style="margin: 0;">⚡ Quick but powerful probabilistic classifier</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("Naive Bayes model..."):
                        X = data.drop(columns=['Class'])
                        y = data['Class']
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, stratify=y, random_state=42
                        )
                        pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('nb', GaussianNB()),
                        ])
                        pipeline.fit(X_train, y_train)
                        y_train_pred = pipeline.predict(X_train)
                        y_test_pred = pipeline.predict(X_test)
                        
                        # Store results
                        st.session_state.results["Naive Bayes"] = {
                            "y_test": y_test,
                            "y_pred": y_test_pred,
                            "train_metrics": classification_report(y_train, y_train_pred, output_dict=True),
                            "test_metrics": classification_report(y_test, y_test_pred, output_dict=True)

                        }
                        
                        # Display metrics in modern cards
                        st.markdown("#### 🏋‍♂ Training Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        metrics = st.session_state.results["Naive Bayes"]["train_metrics"]
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.2%}", "Train")
                        with col2:
                            st.metric("Precision", f"{metrics['1']['precision']:.2%}", "Train")
                        with col3:
                            st.metric("Recall", f"{metrics['1']['recall']:.2%}", "Train")
                        with col4:
                            st.metric("F1-Score", f"{metrics['1']['f1-score']:.2%}", "Train")
                        
                        # Display metrics in modern cards - Testing
                        st.markdown("#### 🧪 Testing Performance")
                        col1, col2, col3, col4 = st.columns(4)
                        metrics = st.session_state.results["Naive Bayes"]["test_metrics"]
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.2%}", "Test")
                        with col2:
                            st.metric("Precision", f"{metrics['1']['precision']:.2%}", "Test")
                        with col3:
                            st.metric("Recall", f"{metrics['1']['recall']:.2%}", "Test")
                        with col4:
                            st.metric("F1-Score", f"{metrics['1']['f1-score']:.2%}", "Test")
                        
                        # Visualizations
                        st.markdown("### 📊 Model Visualizations")
                        fig = plot_confusion_matrix(y_test, y_test_pred, 'Naive Bayes Confusion Matrix')
                        st.plotly_chart(fig, use_container_width=True)
        
        # KNN (updated similarly)
            elif model_name == "KNN":
                with st.expander("🧠 K-Nearest Neighbors", expanded=True):
                    st.markdown("""
                <div class="custom-container">
                    <h3 style="margin: 0;">🔍 Instance-based learning with spatial reasoning</h3>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner(" KNN model..."):
                    X = data.iloc[:, :-1].values
                    y = data.iloc[:, -1].values
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.25, random_state=0
                    )
                    knn_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))
                    ])
                    knn_pipeline.fit(X_train, y_train)
                    y_train_pred = knn_pipeline.predict(X_train)
                    y_test_pred = knn_pipeline.predict(X_test)
                    
                    # Store results
                    st.session_state.results["KNN"] = {
                        "y_test": y_test,
                        "y_pred": y_test_pred,
                        "train_metrics": classification_report(y_train, y_train_pred, output_dict=True),
                        "test_metrics": classification_report(y_test, y_test_pred, output_dict=True),
                        "pipeline": knn_pipeline
                    }
                    
                    # Display metrics - Training
                    st.markdown("#### 🏋‍♂ Training Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = st.session_state.results["KNN"]["train_metrics"]
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}", "Train")
                    with col2:
                        st.metric("Precision", f"{metrics['1']['precision']:.2%}", "Train")
                    with col3:
                        st.metric("Recall", f"{metrics['1']['recall']:.2%}", "Train")
                    with col4:
                        st.metric("F1-Score", f"{metrics['1']['f1-score']:.2%}", "Train")
                    
                    # Display metrics - Testing
                    st.markdown("#### 🧪 Testing Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = st.session_state.results["KNN"]["test_metrics"]
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}", "Test")
                    with col2:
                        st.metric("Precision", f"{metrics['1']['precision']:.2%}", "Test")
                    with col3:
                        st.metric("Recall", f"{metrics['1']['recall']:.2%}", "Test")
                    with col4:
                        st.metric("F1-Score", f"{metrics['1']['f1-score']:.2%}", "Test")
                    
                    # Visualizations
                    st.markdown("### 📊 Model Visualizations")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_confusion_matrix(y_test, y_test_pred, 'KNN Confusion Matrix')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # PCA Visualization (same as before)
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X)
                        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
                            X_pca, y, test_size=0.25, random_state=0
                        )
                        sc_pca = StandardScaler()
                        X_train_pca = sc_pca.fit_transform(X_train_pca)
                        X_test_pca = sc_pca.transform(X_test_pca)
                        classifier_pca = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
                        classifier_pca.fit(X_train_pca, y_train_pca)
                        fig = px.scatter(
                            x=X_test_pca[:, 0], 
                            y=X_test_pca[:, 1], 
                            color=y_test_pca,
                            color_discrete_map={0: '#00cc96', 1: '#ef553b'},
                            labels={'color': 'Class'},
                            title='KNN Decision Boundary (PCA Reduced)'
                        )
                        fig.update_traces(
                            marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
                            selector=dict(mode='markers')
                        )
                        fig.update_layout(
                            xaxis_title='Principal Component 1',
                            yaxis_title='Principal Component 2',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
        
                        st.plotly_chart(fig, use_container_width=True)


        # Similarly update AdaBoost and Decision Tree implementations
            elif model_name=="AdaBoost":
                with st.expander("🚀 AdaBoost Classifier", expanded=True):
                    st.markdown("""
                <div class="custom-container">
                    <h3 style="margin: 0;">🔥 Adaptive Boosting for improved performance</h3>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("AdaBoost model..."):
                    X = data.drop(columns=['Class'])
                    y = data['Class']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, stratify=y, random_state=42
                    )
                    
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('ada', AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(max_depth=1),
                            n_estimators=25,
                            learning_rate=1.0,
                            random_state=42
                        )),
                    ])
                    pipeline.fit(X_train, y_train)
                    
                    # Get predictions for both train and test
                    y_train_pred = pipeline.predict(X_train)
                    y_test_pred = pipeline.predict(X_test)
                    
                    # Store results
                    st.session_state.results["AdaBoost"] = {
                        "y_test": y_test,
                        "y_pred": y_test_pred,
                        "train_metrics": classification_report(y_train, y_train_pred, output_dict=True),
                        "test_metrics": classification_report(y_test, y_test_pred, output_dict=True)
                    }
                    
                    # Display metrics - Training
                    st.markdown("#### 🏋‍♂ Training Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = st.session_state.results["AdaBoost"]["train_metrics"]
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}", "Train")
                    with col2:
                        st.metric("Precision", f"{metrics['1']['precision']:.2%}", "Train")
                    with col3:
                        st.metric("Recall", f"{metrics['1']['recall']:.2%}", "Train")
                    with col4:
                        st.metric("F1-Score", f"{metrics['1']['f1-score']:.2%}", "Train")
                    
                    # Display metrics - Testing
                    st.markdown("#### 🧪 Testing Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = st.session_state.results["AdaBoost"]["test_metrics"]
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}", "Test")
                    with col2:
                        st.metric("Precision", f"{metrics['1']['precision']:.2%}", "Test")
                    with col3:
                        st.metric("Recall", f"{metrics['1']['recall']:.2%}", "Test")
                    with col4:
                        st.metric("F1-Score", f"{metrics['1']['f1-score']:.2%}", "Test")
                    
                    # Visualizations
                    st.markdown("### 📊 Model Visualizations")
                    fig = plot_confusion_matrix(y_test, y_test_pred, 'AdaBoost Confusion Matrix')
                    st.plotly_chart(fig, use_container_width=True)
                    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
                    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=np.abs(y_test - y_test_pred),
                            mode='lines+markers',
                            name='Absolute Error',
                            line=dict(color='orange')
                        ))
                        fig.update_layout(
                            title='Prediction Error Plot',
                            yaxis_title='Error',
                            xaxis_title='Sample Index',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                        auc_score = roc_auc_score(y_test, y_test_proba)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines', name=f'ROC Curve (AUC={auc_score:.2f})',
                            line=dict(color='lime')
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines', line=dict(dash='dash'), showlegend=False
                        ))
                        fig.update_layout(
                            title='ROC Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)



            elif model_name == "Decision Tree":
                with st.expander("🌳 Decision Tree Classifier", expanded=True):
                    st.markdown("""
                <div class="custom-container">
                    <h3 style="margin: 0;">🌿 Simple tree-based model with interpretable decisions</h3>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner("Decision Tree model..."):
                    X = data.drop(columns=['Class'])
                    y = data['Class']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, stratify=y, random_state=100
                    )
                    
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('dt', DecisionTreeClassifier(
                            criterion='gini',
                            max_depth=4,
                            class_weight={0: 1, 1: 3},
                            random_state=100
                        )),
                    ])
                    pipeline.fit(X_train, y_train)
                    
                    # Get predictions for both train and test
                    y_train_pred = pipeline.predict(X_train)
                    y_test_pred = pipeline.predict(X_test)
                    
                    # Store results
                    st.session_state.results["Decision Tree"] = {
                        "y_test": y_test,
                        "y_pred": y_test_pred,
                        "train_metrics": classification_report(y_train, y_train_pred, output_dict=True),
                        "test_metrics": classification_report(y_test, y_test_pred, output_dict=True)
                    }
                    
                    # Display metrics - Training
                    st.markdown("#### 🏋‍♂ Training Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = st.session_state.results["Decision Tree"]["train_metrics"]
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}", "Train")
                    with col2:
                        st.metric("Precision", f"{metrics['1']['precision']:.2%}", "Train")
                    with col3:
                        st.metric("Recall", f"{metrics['1']['recall']:.2%}", "Train")
                    with col4:
                        st.metric("F1-score", f"{metrics['1']['f1-score']:.2%}", "Train")
                    
                    # Display metrics - Testing
                    st.markdown("#### 🧪 Testing Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = st.session_state.results["Decision Tree"]["test_metrics"]
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}", "Test")
                    with col2:
                        st.metric("Precision", f"{metrics['1']['precision']:.2%}", "Test")
                    with col3:
                        st.metric("Recall", f"{metrics['1']['recall']:.2%}", "Test")
                    with col4:
                        st.metric("F1-score", f"{metrics['1']['f1-score']:.2%}", "Test")
                    
                    # Visualizations
                    st.markdown("### 📊 Model Visualizations")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_confusion_matrix(y_test, y_test_pred, 'Decision Tree Confusion Matrix')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Plot the decision tree
                        fig, ax = plt.subplots(figsize=(12, 8))
                        plot_tree(
                            pipeline.named_steps['dt'],
                            filled=True,
                            rounded=True,
                            feature_names=X.columns,
                            class_names=['Genuine', 'Fraud'],
                            ax=ax
                        )
                        plt.title("Decision Tree Visualization")
                        st.pyplot(fig)

# Update the Performance Dashboard tab to show both train and test metrics
with tab2:
    if not models_to_run:
        st.info("ℹ Please select at least one model from the sidebar to view results")
    elif len(models_to_run) < 2:
        st.info("ℹ Please select at least two models from the sidebar to compare")
    else:
        st.markdown("""
        <div class="custom-container">
            <h2 style="margin: 0; color: #6e45e2;">📊 MODEL PERFORMANCE DASHBOARD</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create comparison metrics tables for both train and test
        train_comparison_data = []
        test_comparison_data = []
        
        for model_name in st.session_state.results:
            train_metrics = st.session_state.results[model_name]["train_metrics"]
            test_metrics = st.session_state.results[model_name]["test_metrics"]
            
            train_comparison_data.append({
                "Model": model_name,
                "Accuracy": train_metrics["accuracy"],
                "Precision (Fraud)": train_metrics["1"]["precision"],
                "Recall (Fraud)": train_metrics["1"]["recall"],
                "F1-Score (Fraud)": train_metrics["1"]["f1-score"]
            })
            
            test_comparison_data.append({
                "Model": model_name,
                "Accuracy": test_metrics["accuracy"],
                "Precision (Fraud)": test_metrics["1"]["precision"],
                "Recall (Fraud)": test_metrics["1"]["recall"],
                "F1-Score (Fraud)": test_metrics["1"]["f1-score"]
            })
        
        train_comparison_df = pd.DataFrame(train_comparison_data)
        test_comparison_df = pd.DataFrame(test_comparison_data)
        
        # Training metrics comparison
        st.markdown("### 🏋‍♂ Training Performance Comparison")
        st.dataframe(
            train_comparison_df.style.format({
                'Accuracy': '{:.2%}',
                'Precision (Fraud)': '{:.2%}',
                'Recall (Fraud)': '{:.2%}',
                'F1-Score (Fraud)': '{:.2%}'
            }).background_gradient(cmap='Greens'),
            use_container_width=True
        )
        
        # Testing metrics comparison
        st.markdown("### 🧪 Testing Performance Comparison")
        st.dataframe(
            test_comparison_df.style.format({
                'Accuracy': '{:.2%}',
                'Precision (Fraud)': '{:.2%}',
                'Recall (Fraud)': '{:.2%}',
                'F1-Score (Fraud)': '{:.2%}'
            }).background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        # Interactive comparison chart for test results
        st.markdown("### 📈 Test Performance Comparison")
        fig = plot_metrics_comparison(test_comparison_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices comparison (unchanged)
        st.markdown("### 🧩 Confusion Matrices Comparison")
        cols = st.columns(len(st.session_state.results))
        for i, model_name in enumerate(st.session_state.results):
            with cols[i]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05);
                            border-radius: 12px;
                            padding: 15px;
                            border: 1px solid rgba(255,255,255,0.1);
                            margin-bottom: 20px;">
                    <h4>{model_name}</h4>
                </div>
                """, unsafe_allow_html=True)
                y_test = st.session_state.results[model_name]["y_test"]
                y_pred = st.session_state.results[model_name]["y_pred"]
                fig = plot_confusion_matrix(y_test, y_pred, f'{model_name} Confusion Matrix')
                st.plotly_chart(fig, use_container_width=True)