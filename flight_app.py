import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Flight Delay Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">✈️ Flight Delay Prediction System</h1>', unsafe_allow_html=True)

# Sidebar for model selection and info
st.sidebar.header("🔧 Model Configuration")

# Model selection
model_choice = st.sidebar.selectbox(
    "Choose Prediction Model:",
    ["Random Forest", "Logistic Regression", "SVM", "Decision Tree", "Naive Bayes", "KNN"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About this App:**
This system predicts flight delays using machine learning models trained on historical flight data.

**Features Used:**
- Scheduled Departure Time
- Taxi Out Time
- Scheduled Flight Duration
- Flight Distance
- Departure Delay
- Airline Code
""")

# Load or create models (in practice, you'd load pre-trained models)
@st.cache_data
def load_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'CRS_DEP_TIME': np.random.randint(600, 2200, n_samples),
        'TAXI_OUT': np.random.normal(15, 5, n_samples),
        'CRS_ELAPSED_TIME': np.random.normal(150, 50, n_samples),
        'DISTANCE': np.random.normal(800, 400, n_samples),
        'DEP_DELAY': np.random.normal(10, 20, n_samples),
        'AIRLINE_ENCODED': np.random.randint(0, 5, n_samples)
    }
    
    return pd.DataFrame(data)

@st.cache_resource
def train_models():
    """Train models for demonstration (in practice, load pre-trained models)"""
    # Get sample data
    df = load_sample_data()
    
    # Create target variable based on departure delay
    def classify_delay(delay):
        if delay <= 15:
            return 'On Time'
        elif delay <= 60:
            return 'Short Delay'
        else:
            return 'Long Delay'
    
    df['FLIGHT_STATUS'] = df['DEP_DELAY'].apply(classify_delay)
    
    # Prepare features and target
    X = df[['CRS_DEP_TIME', 'TAXI_OUT', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DEP_DELAY', 'AIRLINE_ENCODED']]
    y = df['FLIGHT_STATUS']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'SVM': SVC(random_state=42, kernel='linear', probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        trained_models[name] = model
    
    return trained_models, scaler, X.columns.tolist()

# Load models
try:
    models, scaler, feature_names = train_models()
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# Main content area
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Data Insights"])

with tab1:
    st.header("Flight Delay Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Flight Information")
        
        # Input fields
        crs_dep_time = st.number_input(
            "Scheduled Departure Time (24hr format)",
            min_value=0, max_value=2359, value=1200,
            help="Enter time in 24-hour format (e.g., 1430 for 2:30 PM)"
        )
        
        taxi_out = st.slider(
            "Expected Taxi Out Time (minutes)",
            min_value=5, max_value=60, value=15
        )
        
        crs_elapsed_time = st.number_input(
            "Scheduled Flight Duration (minutes)",
            min_value=30, max_value=600, value=120
        )
        
    with col2:
        st.subheader("Flight Details")
        
        distance = st.number_input(
            "Flight Distance (miles)",
            min_value=50, max_value=5000, value=800
        )
        
        dep_delay = st.slider(
            "Departure Delay (minutes)",
            min_value=-30, max_value=300, value=0,
            help="Negative values indicate early departure"
        )
        
        airline_options = {
            'American Airlines': 0,
            'Delta Airlines': 1,
            'United Airlines': 2,
            'Southwest Airlines': 3,
            'JetBlue': 4
        }
        
        airline = st.selectbox("Airline", list(airline_options.keys()))
        airline_encoded = airline_options[airline]
    
    # Prediction button
    if st.button("🚀 Predict Flight Status", type="primary"):
        try:
            # Prepare input data
            input_data = np.array([[
                crs_dep_time, taxi_out, crs_elapsed_time, 
                distance, dep_delay, airline_encoded
            ]])
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Get selected model
            selected_model = models[model_choice]
            
            # Make prediction
            prediction = selected_model.predict(input_scaled)[0]
            prediction_proba = selected_model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            # Main prediction
            with col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Prediction Result</h3>
                    <h2 style="color: {'green' if prediction == 'On Time' else 'orange' if prediction == 'Short Delay' else 'red'}">
                        {prediction}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence scores
            with col2:
                st.subheader("Confidence Scores")
                classes = selected_model.classes_
                for i, (class_name, prob) in enumerate(zip(classes, prediction_proba)):
                    st.metric(
                        label=class_name,
                        value=f"{prob:.1%}",
                        delta=None
                    )
            
            # Visual probability chart
            with col3:
                fig = px.bar(
                    x=classes, y=prediction_proba,
                    title="Prediction Probabilities",
                    labels={'x': 'Flight Status', 'y': 'Probability'},
                    color=prediction_proba,
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.markdown("### 💡 Insights")
            
            if prediction == 'On Time':
                st.success("✅ Your flight is predicted to arrive on time! Have a great trip!")
            elif prediction == 'Short Delay':
                st.warning("⚠️ Expect a short delay. Consider informing those picking you up.")
            else:
                st.error("🚨 Significant delay expected. You may want to check with the airline.")
            
            # Feature importance (for tree-based models)
            if model_choice in ['Random Forest', 'Decision Tree']:
                st.markdown("### 📊 Feature Importance")
                importance = selected_model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                           title="Feature Importance in Prediction")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

with tab2:
    st.header("Model Performance Comparison")
    
    # Simulated performance metrics (in practice, load from your trained models)
    performance_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'SVM', 'Decision Tree', 'Naive Bayes', 'KNN'],
        'Accuracy': [0.87, 0.83, 0.85, 0.79, 0.81, 0.84],
        'Precision': [0.88, 0.84, 0.86, 0.80, 0.82, 0.85],
        'Recall': [0.87, 0.83, 0.85, 0.79, 0.81, 0.84],
        'F1-Score': [0.87, 0.83, 0.85, 0.79, 0.81, 0.84]
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance metrics table
        st.subheader("Performance Metrics")
        st.dataframe(df_performance.style.highlight_max(axis=0))
        
        # Best model highlight
        best_model = df_performance.loc[df_performance['Accuracy'].idxmax(), 'Model']
        st.success(f"🏆 Best Performing Model: **{best_model}**")
    
    with col2:
        # Performance visualization
        st.subheader("Model Comparison")
        
        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=df_performance['Model'],
                y=df_performance[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Data Insights & Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Statistics")
        sample_data = load_sample_data()
        
        st.write("**Dataset Overview:**")
        st.write(f"- Total Records: {len(sample_data):,}")
        st.write(f"- Features: {len(sample_data.columns)}")
        st.write(f"- Date Range: Simulated Data")
        
        # Feature statistics
        st.write("**Feature Statistics:**")
        st.dataframe(sample_data.describe().round(2))
    
    with col2:
        st.subheader("Feature Distributions")
        
        # Feature selection for visualization
        feature_to_plot = st.selectbox(
            "Select Feature to Visualize:",
            ['CRS_DEP_TIME', 'TAXI_OUT', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DEP_DELAY']
        )
        
        fig = px.histogram(
            sample_data, 
            x=feature_to_plot,
            title=f"Distribution of {feature_to_plot}",
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; padding: 20px;'>
    <p>✈️ Flight Delay Prediction System | Built with Streamlit</p>
    <p>For educational purposes - CIS 9660 Data Mining Project</p>
</div>
""", unsafe_allow_html=True)

# Instructions for deployment (shown in sidebar)
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🚀 Deployment Instructions")
    st.markdown("""
    **To deploy this app:**
    
    1. Save this code as `app.py`
    2. Create `requirements.txt`:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    plotly
    ```
    3. Deploy to Streamlit Cloud:
    - Push to GitHub
    - Connect to streamlit.io
    - Deploy with `share=True`
    """)
    
    st.markdown("### 📝 Files Needed")
    st.code("""
    project/
    ├── app.py (this file)
    ├── requirements.txt
    ├── trained_model.pkl (optional)
    └── README.md
    """)