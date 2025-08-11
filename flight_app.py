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
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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
                "flights_sample_3m.csv")

    df_processed = df.copy()

    features_to_fill = ['CRS_ELAPSED_TIME', 'DEP_TIME', 'DEP_DELAY', 'ARR_TIME', 
                        'ARR_DELAY', 'ELAPSED_TIME', 'AIR_TIME', 'DELAY_DUE_CARRIER', 
                        'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 
                        'DELAY_DUE_LATE_AIRCRAFT']
    df_processed[features_to_fill] = df_processed[features_to_fill].fillna(0)

    features_to_drop = ['TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON','TAXI_IN', 'CANCELLATION_CODE']
    df_processed.drop(features_to_drop, axis=1, inplace=True)

    return df_processed

@st.cache_data
def sample_data():
    df = load_kaggle_data()
    df = df.sample(n=10000, random_state=42)
    df = df.dropna(subset=['ARR_DELAY', 'CANCELLED'])
    df['MONTH'] = pd.to_datetime(df['FL_DATE']).dt.month
    df['FLIGHT_STATUS'] = np.where(df['ARR_DELAY'] > 15, 'Delayed', 'Not Delayed')
    df = df[['CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 'MONTH', 'AIRLINE', 'DEST', 'ARR_DELAY', 'CANCELLED', 'FLIGHT_STATUS']]
    return df

@st.cache_resource
def train_models():
    df = sample_data()
    X = df[['CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 'MONTH', 'AIRLINE', 'DEST']]
    X = pd.get_dummies(X, columns=['MONTH', 'AIRLINE', 'DEST'], drop_first=True)
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
models, scaler, feature_names, df_training = train_models()

col1, col2 = st.columns(2)
with col1:
    dep_time_input = st.time_input("Departure Time (HH:MM)", value=datetime.strptime("08:00", "%H:%M").time())
    crs_dep_time = int(dep_time_input.strftime("%H%M"))
    airline = st.selectbox("Airline", ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK'])
with col2:
    month = st.selectbox("Flight Month (1–12)", list(range(1, 13)))
    destination = st.selectbox("Destination Airport Code", ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'SFO', 'SEA'])

if st.button("Predict Flight Status with ALL Models", type="primary", use_container_width=True):
    try:
        input_dict = {
            'CRS_DEP_TIME': crs_dep_time,
            'CRS_ELAPSED_TIME': 90,
            f'MONTH_{month}': 1,
            f'AIRLINE_{airline}': 1,
            f'DEST_{destination}': 1
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_df)

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
                color = 'green' if prediction == 'Not Delayed' else 'red'
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

        if most_common == 'Not Delayed':
            st.success("Your flight is expected to be on time or only slightly delayed.")
        elif most_common == 'Delayed':
            st.error("Your flight is expected to be significantly delayed.")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

st.markdown("---")
st.header("Flight Delay Trends")


monthly_max = df_training.loc[df_training.groupby('MONTH')['ARR_DELAY'].idxmax()][['MONTH', 'ARR_DELAY', 'AIRLINE']]
monthly_max['ARR_DELAY'] = monthly_max['ARR_DELAY'].astype(int)
monthly_max.rename(columns={'ARR_DELAY': 'Longest Delay (mins)'}, inplace=True)

st.subheader("Longest Delay by Month and Airline")
styled_table = monthly_max.style.set_properties(subset=['MONTH', 'Longest Delay (mins)'], **{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])]
st.write(styled_table.set_table_styles([{'selector': 'th.col_heading.level0', 'props': [('display', 'none')]}]))

# Bar chart: x = airline, y = delay prob, color = month
delay_chart = df_training.groupby(['AIRLINE', 'MONTH'])['FLIGHT_STATUS'].value_counts(normalize=True).unstack().fillna(0).reset_index()[['AIRLINE', 'MONTH', 'Delayed']]
fig_bar = px.bar(delay_chart, x='AIRLINE', y='Delayed', color='MONTH', barmode='group',
                 title="Delay Probability by Airline and Month",
                 labels={'Delayed': 'Probability of Delay', 'AIRLINE': 'Airline'})
fig_bar.update_traces(hoverinfo='none', hovertemplate=None)
fig_bar.update_layout(clickmode='none')
st.plotly_chart(fig_bar, use_container_width=True)
