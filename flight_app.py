
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

st.title("Flight Delay Predictor")

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
def sample_data(n=10000, seed=42):
    df = load_kaggle_data()
    df = df.sample(n=n, random_state=seed)
    df = df.dropna(subset=['ARR_DELAY', 'CANCELLED'])
    df['MONTH'] = pd.to_datetime(df['FL_DATE']).dt.month
    df['FLIGHT_STATUS'] = np.where(df['ARR_DELAY'] > 15, 'Delayed', 'Not Delayed')
    df = df[['CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 'MONTH', 'AIRLINE', 'DEST', 'ARR_DELAY', 'CANCELLED', 'FLIGHT_STATUS']]
    return df

@st.cache_resource
def prepare_train_test():
    df = sample_data()
    X = df[['CRS_DEP_TIME', 'CRS_ELAPSED_TIME', 'MONTH', 'AIRLINE', 'DEST']]
    X = pd.get_dummies(X, columns=['MONTH', 'AIRLINE', 'DEST'], drop_first=True)
    y = df['FLIGHT_STATUS']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = X.columns.tolist()
    return (df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names)

def get_models():
    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, solver='liblinear'),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10),
        'SVM (Linear)': SVC(random_state=42, kernel='linear', probability=True),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5)
    }


tab_pred, tab_eval, tab_kmeans = st.tabs([
    "Predict (Ensemble Vote)",
    "Training & Evaluation (Reports, CV, Weighted Avgs)",
    "K-Means Elbow"
])

(df_training,
 X_train, X_test, y_train, y_test,
 X_train_scaled, X_test_scaled, scaler, feature_names) = prepare_train_test()
models = get_models()

with tab_pred:
    st.subheader("Predict Flight Status")
    col1, col2 = st.columns(2)
    with col1:
        dep_time_input = st.time_input("Departure Time (HH:MM)", value=datetime.strptime("08:00", "%H:%M").time())
        crs_dep_time = int(dep_time_input.strftime("%H%M"))
        airline = st.selectbox("Airline", sorted(df_training['AIRLINE'].unique().tolist()))
    with col2:
        month = st.selectbox("Flight Month (1–12)", list(range(1, 13)))
        destination = st.selectbox("Destination Airport Code", sorted(df_training['DEST'].unique().tolist()))

    if st.button("Predict", type="primary", use_container_width=True):
      
        input_dict = {'CRS_DEP_TIME': crs_dep_time, 'CRS_ELAPSED_TIME': 90}

        input_dict.update({f'MONTH_{month}': 1, f'AIRLINE_{airline}': 1, f'DEST_{destination}': 1})
        input_df = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_df)

        all_predictions = []
        for model in models.values():
            model.fit(X_train_scaled, y_train) 
            pred = model.predict(input_scaled)[0]
            all_predictions.append(pred)


        most_common = max(set(all_predictions), key=all_predictions.count)
        if most_common == 'Not Delayed':
            st.success("Your flight is expected to be on time or only slightly delayed.")
        else:
            st.error("Your flight is expected to be significantly delayed.")

with tab_eval:
    st.subheader("Model Evaluation")
    st.write("Includes 5-fold CV (training set) and classification reports (test set).")


    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained[name] = model

 
    cv_scores = {}
    for name, model in trained.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_scores[name] = float(np.mean(scores))

    cv_df = pd.DataFrame({'Model': list(cv_scores.keys()), 'CV_Accuracy': list(cv_scores.values())})
    cv_fig = px.bar(cv_df, x='Model', y='CV_Accuracy', title="5-Fold Cross-Validation (Mean Accuracy)")
    cv_fig.update_layout(yaxis=dict(tickformat=".0%"))
    st.plotly_chart(cv_fig, use_container_width=True)


    reports = []
    metrics_for_weighted_avg = []
    for name, model in trained.items():
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        row = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Weighted Precision': report['weighted avg']['precision'],
            'Weighted Recall': report['weighted avg']['recall'],
            'Weighted F1': report['weighted avg']['f1-score']
        }
        reports.append(row)


        metrics_for_weighted_avg.append({
            'Model': name, 'Metric': 'Accuracy', 'Value': row['Accuracy']
        })
        metrics_for_weighted_avg.append({
            'Model': name, 'Metric': 'Precision', 'Value': row['Weighted Precision']
        })
        metrics_for_weighted_avg.append({
            'Model': name, 'Metric': 'Recall', 'Value': row['Weighted Recall']
        })
        metrics_for_weighted_avg.append({
            'Model': name, 'Metric': 'F1', 'Value': row['Weighted F1']
        })

    report_df = pd.DataFrame(reports).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    st.dataframe(report_df.style.format({
        'Accuracy': '{:.3f}', 'Weighted Precision': '{:.3f}', 'Weighted Recall': '{:.3f}', 'Weighted F1': '{:.3f}'
    }), use_container_width=True)


    long_df = pd.DataFrame(metrics_for_weighted_avg)
    fig = px.bar(long_df, x='Model', y='Value', color='Metric', barmode='group',
                 title="Model vs. Weighted Avg Metrics (Test Set)")
    fig.update_layout(yaxis=dict(title='Score', tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)

with tab_kmeans:
    st.subheader("Elbow Method for K-Means")
    st.write("Runs on the **training features** to find a reasonable k.")

    k_range = st.slider("Choose k range", min_value=2, max_value=20, value=(2, 14))
    inertia = []
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        inertia.append(kmeans.inertia_)

    elbow_fig = go.Figure(data=go.Scatter(
        x=list(range(k_range[0], k_range[1] + 1)),
        y=inertia,
        mode='lines+markers'
    ))
    elbow_fig.update_layout(
        title="Elbow Method (Sum of Squared Distances vs k)",
        xaxis_title="k",
        yaxis_title="Sum of Squared Distances"
    )
    st.plotly_chart(elbow_fig, use_container_width=True)
