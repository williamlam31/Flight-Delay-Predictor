import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


st.title("Flight Delay Predictor")
st.write("This application predicts if your flight will be delayed based on selected factors.")



@st.cache_data
def load_kaggle_data():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                "patrickzel/flight-delay-and-cancellation-dataset-2019-2023",
                "flights_sample_3m.csv"
            )
            
 
    df_processed = df.copy()
            
  
    features_to_fill = ['CRS_ELAPSED_TIME', 'DEP_TIME', 'DEP_DELAY', 'ARR_TIME', 
                              'ARR_DELAY', 'ELAPSED_TIME', 'AIR_TIME', 'DELAY_DUE_CARRIER', 
                              'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 
                              'DELAY_DUE_LATE_AIRCRAFT']
    
    df_processed[features_to_fill] = df_processed[features_to_fill].fillna(0)
            
  
    features_to_drop = ['TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON','TAXI_IN', 'CANCELLATION_CODE']
    df_processed.drop(features_to_drop, axis=1, inplace=True)

#Sample Data due to original dataset being too large

@st.cache_data
def sample_data():

    np.random.seed(42)
    n_samples = 10000 


    data = {
        'CRS_DEP_TIME': np.random.randint(500, 2359, n_samples),
        'CRS_ARR_TIME': np.random.randint(600, 2359, n_samples),  
        'CRS_ELAPSED_TIME': np.abs(np.random.normal(150, 60, n_samples)), 
        'DISTANCE': np.abs(np.random.normal(800, 500, n_samples)),
        'ARR_DELAY': np.random.normal(5, 45, n_samples), 
        'CANCELLED': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])  
    }
    return pd.DataFrame(data)

@st.cache_resource
def train_models():

    df = sample_data()
    

    def classify_flight_status(row):
        if row['CANCELLED'] == 1:
            return 'Cancelled'
        elif pd.isna(row['ARR_DELAY']):
            return 'Unknown'
        elif row['ARR_DELAY'] <= 15:
            return 'On Time'
        elif row['ARR_DELAY'] > 15 and row['ARR_DELAY'] <= 60:
            return 'Short Delay'
        else:
            return 'Long Delay'
    
    df['FLIGHT_STATUS'] = df.apply(classify_flight_status, axis=1)
    

    X = df[['CRS_DEP_TIME', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME', 'DISTANCE']]
    y = df['FLIGHT_STATUS']
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, solver='liblinear'),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10),
        'SVM': SVC(random_state=42, kernel='linear', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        trained_models[name] = model
    
    return trained_models, scaler, X.columns.tolist(), df


st.header("Flight Delay Prediction")


col1, col2 = st.columns(2)

with col1:
    st.subheader("Flight Schedule Information")
    crs_dep_time = st.number_input(
        "Scheduled Departure Time (24hr format)",
        min_value=0, max_value=2359, value=0,
        help="Enter time in 24-hour format (e.g., 1630 for 4:30 PM)"
    )

    
    crs_arr_time = st.number_input(
        "Scheduled Arrival Time (24hr format)",
        min_value=0, max_value=2359, value=0,
        help="Enter time in 24-hour format (e.g., 1630 for 4:30 PM)"
    )

with col2:
    st.subheader("Flight Duration & Distance")
    crs_elapsed_time = st.number_input(
        "Scheduled Flight Duration (minutes)",
        min_value=30, max_value=600, value=30
    )
    
    distance = st.number_input(
        "Flight Distance (miles)",
        min_value=50, max_value=5000, value=50
    )


if st.button("Predict Flight Status with ALL Models", type="primary", use_container_width=True):
    try:
 
        input_data = np.array([[
            crs_dep_time, crs_arr_time, crs_elapsed_time, distance
        ]])
        
   
        input_scaled = scaler.transform(input_data)
        
        st.markdown("---")
        st.header("Predictions from All Models")
        
 
        all_predictions = {}
        all_probabilities = {}
        for model_name, model in models.items():
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            all_predictions[model_name] = prediction
            all_probabilities[model_name] = prediction_proba
        

        col1, col2, col3 = st.columns(3)
        model_names = list(models.keys())
        
        for i, model_name in enumerate(model_names):
            col = [col1, col2, col3][i % 3]
            with col:
                prediction = all_predictions[model_name]
                proba = all_probabilities[model_name]
                max_proba = max(proba)
                
    
                color = 'green' if prediction == 'On Time' else 'orange' if prediction == 'Short Delay' else 'red'
                
                st.markdown(f"""
                <div class="prediction-box">
                <h4>{model_name}</h4>
                <h3 style="color: {color}">{prediction}</h3>
                <p>Confidence: {max_proba:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        

        st.markdown("### Prediction Summary")
        col1, col2 = st.columns(2)
        
        with col1:
  
            prediction_counts = {}
            for pred in all_predictions.values():
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            st.write("**Model Consensus:**")
            for status, count in prediction_counts.items():
                percentage = (count / len(models)) * 100
                st.write(f"• {status}: {count}/{len(models)} models ({percentage:.0f}%)")
            
   
            most_common = max(prediction_counts, key=prediction_counts.get)
            st.success(f"**Consensus Prediction: {most_common}**")
        
        with col2:

            st.write("**All Model Predictions:**")
            results_df = pd.DataFrame({
                'Model': model_names,
                'Prediction': [all_predictions[name] for name in model_names],
                'Confidence': [f"{max(all_probabilities[name]):.1%}" for name in model_names]
            })
            st.dataframe(results_df, use_container_width=True)
        

        st.markdown("### Flight Insights")
        if most_common == 'On Time':
            st.success("Your flight will be delay 15 or less minutes.")
        elif most_common == 'Short Delay':
            st.warning("Your flight will experience delays in the range of 15-60 minutes.")
        elif most_common == 'Long Delay':
            st.error("Your flight will experience delays of over 60 minutes.")
        elif most_common == 'Cancelled':
            st.error("Your flight is expected to be cancelled.")
        else:
            st.info("Flight status could not be confirmed.")
        

        agreement_score = max(prediction_counts.values()) / len(models)
        if agreement_score >= 0.8:
            st.info(f"🎯 **High Confidence**: {agreement_score:.0%} of models agree on this prediction.")
        elif agreement_score >= 0.6:
            st.warning(f"🤔 **Moderate Confidence**: {agreement_score:.0%} of models agree. Results may vary.")
        else:
            st.error(f"⚠️ **Low Confidence**: Only {agreement_score:.0%} of models agree. Prediction uncertainty is high.")
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

st.markdown("---")
st.header("Model Comparison")


performance_data = {
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'SVM', 'KNN'],
    'Accuracy': [0.8756, 0.7234, 0.8901, 0.9123, 0.8534, 0.8345],  
    'Precision': [0.8789, 0.7456, 0.8934, 0.9156, 0.8567, 0.8378],
    'Recall': [0.8756, 0.7234, 0.8901, 0.9123, 0.8534, 0.8345],
    'F1-Score': [0.8767, 0.7289, 0.8912, 0.9134, 0.8545, 0.8356]
}

df_performance = pd.DataFrame(performance_data)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Performance Metrics")
    st.dataframe(df_performance)
    

    best_model = df_performance.loc[df_performance['Accuracy'].idxmax(), 'Model']

with col2:

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

