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

        all_predictions = []
        for model in models.values():
            prediction = model.predict(input_scaled)[0]
            all_predictions.append(prediction)

        most_common = max(set(all_predictions), key=all_predictions.count)

        if most_common == 'Not Delayed':
            st.success("Your flight is expected to be on time or only slightly delayed.")
        elif most_common == 'Delayed':
            st.error("Your flight is expected to be significantly delayed.")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

st.markdown("---")
st.header("Flight Delay Trends")

#Table Creation
monthly_max = df_training.loc[df_training.groupby('MONTH')['ARR_DELAY'].idxmax()][['MONTH', 'ARR_DELAY', 'AIRLINE']]
monthly_max['ARR_DELAY'] = monthly_max['ARR_DELAY'].astype(int)
monthly_max.rename(columns={'ARR_DELAY': 'Longest Delay (mins)'}, inplace=True)

st.subheader("Longest Delay by Month and Airline")
monthly_max = monthly_max.reset_index(drop=True)
styled_table = monthly_max.style.set_properties(subset=['MONTH', 'Longest Delay (mins)'], **{'text-align': 'center'})
st.write(styled_table.hide(axis='index').to_html(), unsafe_allow_html=True)

#Barchart
flight_counts = df_training['AIRLINE'].value_counts()
top_airlines = flight_counts.head(10).index
filtered_df = df_training[df_training['AIRLINE'].isin(top_airlines)]

flight_counts_top = filtered_df['AIRLINE'].value_counts().sort_index()
delay_rates_top = (
    filtered_df.groupby('AIRLINE')['FLIGHT_STATUS']
    .value_counts(normalize=True)
    .unstack()
    .get('Delayed', pd.Series(0, index=top_airlines))
)

combined_df = pd.DataFrame({
    'AIRLINE': flight_counts_top.index,
    'Flight Count': flight_counts_top.values,
    'Delay Rate': delay_rates_top.values
})

fig = px.bar(combined_df, x='AIRLINE', y='Flight Count',
             title="Top 10 Airlines: Flight Volume vs Delay Rate")

fig.add_scatter(x=combined_df['AIRLINE'], y=combined_df['Delay Rate'],
                mode='lines+markers', name='Delay Rate',
                yaxis='y2')

fig.update_layout(
    yaxis=dict(title='Flight Count'),
    yaxis2=dict(title='Delay Rate', overlaying='y', side='right', tickformat=".0%"),
    legend=dict(x=0.75, y=1.1),
)

st.plotly_chart(fig, use_container_width=True)
